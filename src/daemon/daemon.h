// OffloadDaemon: the centralized GPU tensor-offload runtime daemon.
//
// The daemon is CUDA-FREE. It owns:
//   - fd-backed pinned data arenas (one per NUMA node), NUMA-localized.
//   - one fd-backed control region (slot table) shared with all ranks.
//   - slot allocation + leases, tensor location table, cold store.
//   - background drain (pinned -> pageable/NVMe) and readback workers.
//   - heartbeat monitoring and lease recovery.
//
// Concurrency model: one std::thread per accepted client connection reads a
// request, dispatches to the matching handler (all under mu_), and writes the
// reply. Background worker pools (drain / readback) and a heartbeat monitor run
// as separate threads. Long IO/memcpy in workers is done WITHOUT holding mu_:
// copy the needed slot metadata out under the lock, release, do IO, re-lock to
// commit, exactly per the design sketch.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "offload_abi.h"
#include "completion_ring.h"
#include "canonical.h"
#include "config.h"
#include "control_layout.h"
#include "protocol.h"
#include "protocol_v2.h"

namespace offload {

class OffloadDaemon {
 public:
    explicit OffloadDaemon(const DaemonConfig& cfg);
    ~OffloadDaemon();

    OffloadDaemon(const OffloadDaemon&) = delete;
    OffloadDaemon& operator=(const OffloadDaemon&) = delete;

    // Blocks: runs the accept loop + background workers until request_stop() or
    // a Shutdown RPC. Serves each connection on its own thread.
    void run();

    // Signal the accept loop + workers to exit. Async-signal-safe enough to be
    // called from a signal handler (writes to a self-pipe + sets atomic flag).
    void request_stop();

    // Test / introspection helpers (thread-safe).
    const std::string& socket_path() const { return cfg_.socket_path; }
    uint32_t num_slots() const { return layout_.num_slots; }

 private:
    // ---- arena / control shared memory ----
    struct Arena {
        uint32_t numa_node = 0;
        uint64_t size = 0;
        int fd = -1;             // memfd
        void* base = nullptr;    // daemon-local mmap (MAP_SHARED)
        uint64_t arena_id = 0;   // == index
        uint32_t registration_granularity = 0;
        uint32_t allocation_granularity = 0;
    };

    // ---- per-window slot grouping (for allocation preference) ----
    static constexpr uint32_t kOverflowGpu = 0xFFFFFFFFu;

    struct Window {
        uint32_t numa_node = 0;
        uint32_t gpu = 0;             // kOverflowGpu for overflow window
        bool is_overflow = false;
        std::vector<uint32_t> slot_ids;  // contiguous, in address order
    };

    // ---- session table ----
    struct Session {
        uint32_t rank_id = 0;
        uint64_t rank_epoch = 0;
        uint64_t pid = 0;
        uint32_t gpu_id = 0;
        uint32_t numa_node = 0;
        uint64_t last_heartbeat_ns = 0;
        bool alive = true;
        uint32_t ring_index = 0xFFFFFFFFu;  // assigned completion-ring index
    };

    // ---- lease table ----
    struct Lease {
        uint64_t lease_id = 0;
        uint32_t base_slot = 0;
        uint32_t slot_count = 0;
        uint64_t tensor_id = 0;
        uint64_t version = 0;
        uint32_t owner_rank = 0;
        uint64_t owner_pid = 0;
        uint64_t rank_epoch = 0;
        uint64_t nbytes = 0;
        bool is_prefetch = false;   // readback lease (H2D path) vs offload (D2H)
    };

    // ---- location table ----
    struct Location {
        uint64_t tensor_id = 0;
        uint64_t version = 0;
        ofld_location_kind_t kind = OFLD_LOC_NONE;
        uint32_t slot_id = 0;       // valid when kind == PINNED
        uint64_t cold_ref = 0;      // valid when kind == PAGEABLE / NVME
        uint64_t nbytes = 0;
    };

    // ---- cold store entries ----
    struct ColdEntry {
        ofld_location_kind_t kind = OFLD_LOC_NONE;  // PAGEABLE or NVME
        uint64_t nbytes = 0;
        // pageable:
        void* pageable = nullptr;
        // nvme: one or more stripe files (path + offset + length)
        struct NvmeChunk { std::string path; uint64_t offset; uint64_t length; };
        std::vector<NvmeChunk> nvme_chunks;
    };

    // ---- work queue items ----
    struct DrainJob { uint32_t base_slot; uint32_t slot_count; };
    struct ReadbackJob {
        uint32_t base_slot;
        uint32_t slot_count;
        uint64_t cold_ref;
        uint64_t nbytes;
    };

    // ------------------------------------------------------------------
    // Setup / teardown
    // ------------------------------------------------------------------
    void build_arenas_and_control();
    void build_slot_plan();
    void teardown();

    // ------------------------------------------------------------------
    // Slot / control helpers (call with mu_ held unless noted)
    // ------------------------------------------------------------------
    ofld_slot_entry_t* slot(uint32_t id) { return &slots_[id]; }
    // Store slot state atomically with release semantics (observable by ranks).
    void set_slot_state(uint32_t id, ofld_slot_state_t st);
    uint32_t slot_state(uint32_t id) const;
    void* slot_addr(uint32_t id);  // daemon-local pointer into the arena

    // Contiguous multi-slot allocator. Returns base slot id (>=0) into *base_out
    // and count, or false if no contiguous run of `count` free slots exists in
    // the given window. Marks the run RESERVED_D2H and fills metadata.
    bool try_alloc_in_window(const Window& w, uint32_t count, uint32_t* base_out);

    // Full allocation preference chain. Returns true and fills base/count on
    // success. blocked_out set true when refused for pressure reasons.
    bool allocate_slots(uint32_t gpu_id, uint32_t numa_node, uint64_t nbytes,
                        uint64_t flags, bool for_prefetch,
                        uint32_t* base_out, uint32_t* count_out,
                        bool* blocked_out);

    // Free a contiguous run back to FREE and clear metadata.
    void free_slots(uint32_t base_slot, uint32_t slot_count);

    // Synchronously drain the oldest eligible PINNED_VALID run to relieve
    // pressure. lk must own mu_; it is temporarily unlocked for IO. Returns
    // true if a run was drained + freed.
    bool force_drain_one(std::unique_lock<std::mutex>& lk);

    uint32_t slots_for_bytes(uint64_t nbytes) const;
    uint64_t used_pinned_bytes() const;   // sum of granularity over non-free slots
    double pinned_pressure() const;       // used / total

    // ------------------------------------------------------------------
    // Session / lease / location helpers (mu_ held)
    // ------------------------------------------------------------------
    bool epoch_valid(uint32_t rank_id, uint64_t rank_epoch);
    Lease* find_lease(uint64_t lease_id);
    // Update location only if version >= current latest. Returns true if it was
    // the latest (i.e. applied). Records the completion either way.
    bool update_location_if_latest(uint64_t tensor_id, uint64_t version,
                                   ofld_location_kind_t kind, uint32_t slot_id,
                                   uint64_t cold_ref, uint64_t nbytes);
    uint64_t latest_version(uint64_t tensor_id) const;

    bool should_drain(const ofld_slot_entry_t* s) const;

    // ------------------------------------------------------------------
    // RPC handlers
    // ------------------------------------------------------------------
    RegisterRankResponse handle_register(const RegisterRankRequest& req,
                                         std::vector<int>* out_fds);
    HeartbeatResponse handle_heartbeat(const HeartbeatRequest& req);
    RequestOffloadResponse handle_request_offload(const RequestOffloadRequest& req);
    MarkD2HSubmittedResponse handle_mark_d2h_submitted(const MarkD2HSubmittedRequest& req);
    MarkD2HCompleteResponse handle_mark_d2h_complete(const MarkD2HCompleteRequest& req);
    RequestPrefetchResponse handle_request_prefetch(const RequestPrefetchRequest& req);
    MarkH2DSubmittedResponse handle_mark_h2d_submitted(const MarkH2DSubmittedRequest& req);
    MarkH2DCompleteResponse handle_mark_h2d_complete(const MarkH2DCompleteRequest& req);
    ReleaseLeaseResponse handle_release_lease(const ReleaseLeaseRequest& req);
    LocationQueryResponse handle_query_location(const LocationQueryRequest& req);
    BatchCompleteResponse handle_batch_complete(const BatchCompleteRequest& req);
    GetStatsResponse handle_get_stats(const GetStatsRequest& req);
    ShutdownResponse handle_shutdown(const ShutdownRequest& req);

    // ------------------------------------------------------------------
    // v2 canonical model-state RPC handlers (canonical.cpp). All additive; if
    // cfg_.v2_enable_canonical_store is false they return a disabled error.
    // ------------------------------------------------------------------
    RegisterJobResponse handle_register_job(const RegisterJobRequest& req);
    RequestCanonicalEvictResponse handle_request_canonical_evict(
        const RequestCanonicalEvictRequest& req);
    CommitCanonicalObjectResponse handle_commit_canonical_object(
        const CommitCanonicalObjectRequest& req);
    SealModelVersionResponse handle_seal_model_version(const SealModelVersionRequest& req);
    GetLatestSealedVersionResponse handle_get_latest_sealed_version(
        const GetLatestSealedVersionRequest& req);
    GetManifestResponse handle_get_manifest(const GetManifestRequest& req);
    PullTensorResponse handle_pull_tensor(const PullTensorRequest& req);
    RequestCanonicalRestoreResponse handle_request_canonical_restore(
        const RequestCanonicalRestoreRequest& req);
    ReleaseCanonicalRestoreResponse handle_release_canonical_restore(
        const ReleaseCanonicalRestoreRequest& req);

    // ---- canonical helpers (mu_ held unless noted; canonical.cpp) ----
    static std::string canonical_key_string(const CanonicalTensorKeyWire& k);
    JobRecord* find_job(const JobUID& uid);            // nullptr if absent
    JobRecord* find_job_by_wire(const JobKeyWire& jk);
    CanonicalObject* find_object(uint64_t object_id);
    CanonicalObject* find_object_by_key(const std::string& key_str);
    // Resolve a canonical object's current physical bytes into a host-owned
    // export staging buffer. Sets *ok; may unlock lk for NVMe/pageable IO.
    std::vector<uint8_t> stage_object_for_export(
        std::unique_lock<std::mutex>& lk, CanonicalObject& obj, bool* ok);
    // 128-bit FNV-1a-based content hash (dedup verification + manifest hashing).
    static void hash_bytes(const void* p, uint64_t n, uint64_t* lo, uint64_t* hi);
    // Best-effort per-job pinned quota check.
    bool quota_allows_pinned(const JobRecord& job, uint64_t nbytes) const;
    // Release a canonical object's physical backing + table entry (mu_ held).
    void release_object(CanonicalObject& obj);

    // Shared logic for a single D2H completion (used by MarkD2HComplete and
    // BatchComplete). Returns true if accepted. mu_ held.
    bool apply_d2h_complete(uint32_t rank_id, uint64_t rank_epoch,
                            uint64_t lease_id, uint64_t tensor_id,
                            uint64_t version, uint32_t slot_id,
                            uint64_t complete_seq, bool* out_latest,
                            bool* out_drain_enqueued);
    bool apply_h2d_complete(uint32_t rank_id, uint64_t rank_epoch,
                            uint64_t lease_id, uint64_t tensor_id,
                            uint64_t version, uint32_t slot_id, bool keep_pinned,
                            uint64_t complete_seq);

    // ------------------------------------------------------------------
    // Connection handling
    // ------------------------------------------------------------------
    void accept_loop();
    void serve_connection(int fd);
    void dispatch(int fd, OpCode op, const std::vector<uint8_t>& payload);

    // ------------------------------------------------------------------
    // Background workers
    // ------------------------------------------------------------------
    void drain_worker_loop();
    void readback_worker_loop();
    void heartbeat_monitor_loop();

    // Perform the actual cold write for a slot run (no mu_ held). Returns the
    // cold entry describing where data landed, and its resolved kind.
    ColdEntry do_drain_to_cold(uint32_t base_slot, uint32_t slot_count,
                               uint64_t nbytes, bool* ok);
    // Read cold data back into the pinned slot run (no mu_ held).
    bool do_readback_from_cold(uint32_t base_slot, uint32_t slot_count,
                               const ColdEntry& cold, uint64_t nbytes);

    // NVMe helpers. Two engines: synchronous O_DIRECT pwrite/pread, and an
    // io_uring engine (config nvme.io_engine == "io_uring"). Both share the
    // striping layout and produce/consume identical ColdEntry::NvmeChunk lists.
    bool nvme_write(uint64_t cold_ref, const void* src, uint64_t nbytes,
                    ColdEntry* out);
    bool nvme_read(const ColdEntry& e, void* dst, uint64_t nbytes);
    bool nvme_write_pwrite(const void* src, uint64_t nbytes, ColdEntry* out);
    bool nvme_read_pwrite(const ColdEntry& e, void* dst, uint64_t nbytes);
    bool nvme_write_uring(const void* src, uint64_t nbytes, ColdEntry* out);
    bool nvme_read_uring(const ColdEntry& e, void* dst, uint64_t nbytes);
    bool use_uring_engine() const;
    void ensure_nvme_dir();
    // Directory for stripe i (round-robins over nvme_stripe_dirs if set).
    std::string stripe_dir(uint32_t stripe) const;
    // Per-stripe segment byte sizes for a tensor of nbytes.
    std::vector<uint64_t> stripe_segments(uint64_t nbytes) const;

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    DaemonConfig cfg_;

    // control region
    int control_fd_ = -1;
    void* control_base_ = nullptr;
    ControlLayout layout_{};
    ofld_slot_table_header_t* header_ = nullptr;
    ofld_arena_desc_t* arena_descs_ = nullptr;
    ofld_slot_entry_t* slots_ = nullptr;
    uint64_t daemon_epoch_ = 0;

    std::vector<Arena> arenas_;
    std::vector<Window> windows_;
    // Index: (numa_node, gpu) -> window index; overflow keyed by gpu=kOverflowGpu.
    std::unordered_map<uint64_t, uint32_t> window_index_;
    uint64_t window_key(uint32_t numa, uint32_t gpu) const {
        return (static_cast<uint64_t>(numa) << 32) | gpu;
    }

    // tables (all under mu_)
    std::mutex mu_;
    std::unordered_map<uint32_t, Session> sessions_;      // rank_id -> session
    std::unordered_map<uint64_t, Lease> leases_;          // lease_id -> lease
    std::unordered_map<uint64_t, Location> locations_;    // tensor_id -> latest
    std::unordered_map<uint64_t, ColdEntry> cold_store_;  // cold_ref -> entry
    uint64_t next_lease_id_ = 1;
    uint64_t next_rank_epoch_ = 1;
    uint64_t next_cold_ref_ = 1;

    // ---- v2 canonical model-state tables (all under mu_) ----
    // job identity -> record
    std::unordered_map<JobUID, JobRecord, JobUIDHash> jobs_;
    // object_id -> canonical object
    std::unordered_map<uint64_t, CanonicalObject> objects_;
    // serialized canonical key -> object_id (dedup lookup)
    std::unordered_map<std::string, uint64_t> object_by_key_;
    // (job, role, version) -> manifest
    std::unordered_map<VersionKey, Manifest, VersionKeyHash> manifests_;
    // (job, role) -> latest promoted sealed version (+present flag)
    std::unordered_map<RoleKey, uint64_t, RoleKeyHash> latest_sealed_;
    uint64_t next_job_id_ = 1;
    uint64_t next_object_id_ = 1;
    uint64_t next_launch_epoch_ = 1;

    // drain queue
    std::deque<DrainJob> drain_queue_;
    std::condition_variable drain_cv_;
    std::mutex drain_mu_;

    // readback queue
    std::deque<ReadbackJob> readback_queue_;
    std::condition_variable readback_cv_;
    std::mutex readback_mu_;

    // threads
    std::vector<std::thread> drain_threads_;
    std::vector<std::thread> readback_threads_;
    std::thread heartbeat_thread_;
    std::thread ring_poller_thread_;
    std::vector<std::thread> conn_threads_;
    std::mutex conn_threads_mu_;
    std::unordered_set<int> conn_fds_;   // live connection fds (guarded by conn_threads_mu_)

    // completion-ring state
    uint32_t ring_max_ranks_ = 0;
    uint32_t ring_stride_ = 0;
    uint32_t ring_capacity_ = 0;
    // ring index -> owning rank_id (0xFFFFFFFF = free). Guarded by mu_.
    std::vector<uint32_t> ring_owner_;
    void ring_poller_loop();
    // Apply one completion popped from a ring (same checks as the RPC path).
    void apply_ring_completion(uint32_t rank_id, uint64_t rank_epoch,
                               const ofld_completion_entry_t& e);

    // lifecycle
    int listen_fd_ = -1;
    // Optional TCP control endpoint for remote rollout engines (v2). -1 unless
    // cfg_.v2_control_tcp_port != 0. Serves the SAME dispatch as the UDS, but
    // never passes fds (SCM_RIGHTS is UDS-only), so remote clients use only the
    // canonical/rollout RPCs (RegisterRank returns fds → local ranks only).
    int tcp_listen_fd_ = -1;
    int stop_pipe_[2] = {-1, -1};   // self-pipe: [0]=read, [1]=write
    std::atomic<bool> running_{false};
    std::atomic<bool> workers_running_{false};

    // nvme striping state
    uint32_t nvme_stripe_count_ = 4;
    std::atomic<uint64_t> nvme_file_offset_{0};  // append cursor per stripe below
    std::vector<std::atomic<uint64_t>> nvme_stripe_offsets_;
    std::mutex nvme_mu_;
};

}  // namespace offload
