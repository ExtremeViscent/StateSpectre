// Rank-side OffloadAgent implementation (the Impl behind agent.h).
//
// This translation unit is the ONLY place the agent touches CUDA. agent.h keeps
// device pointers as uint64_t and streams as an opaque StreamHandle so the
// Python/pybind boundary never links CUDA symbols. Here we cast those back to
// real cudaStream_t / device pointers and drive the CUDA runtime.
//
// Responsibilities (DESIGN_SPEC §3 rank-side owns): mmap of data/control fds,
// one-off cudaHostRegister per chunk (NEVER per copy — §6/§14.1), D2H/H2D
// cudaMemcpyAsync, CUDA event record/poll, storage invalidation only AFTER the
// D2H event completes (RACE_CONDITIONS §1), restore that never H2Ds from a slot
// that is not PINNED_VALID (RACE_CONDITIONS §10), and RPC completion reporting
// with rank_epoch on every message (RACE_CONDITIONS §6).

#include "agent.h"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "control_layout.h"
#include "completion_ring.h"
#include "metrics.h"
#include "numa_util.h"
#include "offload_abi.h"
#include "offload_canonical_abi.hpp"
#include "protocol.h"
#include "rpc_client.h"
#include "uds.h"
#include "util.h"

namespace offload {

// The v2 AttachCreateAction enum values live in namespace offload_v2 (see
// abi/offload_canonical_abi.hpp). canonical_evict() compares against them.
using namespace offload_v2;

namespace {

constexpr const char* kTag = "agent";

// 128-bit content hash of a host buffer. MUST match the daemon's hash_bytes()
// (canonical.cpp) byte-for-byte so hash-verified dedup compares equal across
// the wire. Two seeded FNV-1a streams.
inline void hash_host_bytes(const void* p, uint64_t n, uint64_t* lo, uint64_t* hi) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h1 = 1469598103934665603ull;
    uint64_t h2 = 1099511628211ull;
    for (uint64_t i = 0; i < n; ++i) {
        h1 = (h1 ^ b[i]) * 1099511628211ull;
        h2 = (h2 ^ b[i]) * 1099511628211ull;
        h2 = (h2 << 1) | (h2 >> 63);
    }
    *lo = h1;
    *hi = h2;
}

// Throwing CUDA error check. agent.cpp only calls the CUDA *runtime API* (no
// kernels), so plain g++ + libcudart is sufficient.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error at " #expr ": ") + \
                                     cudaGetErrorString(_err));                \
        }                                                                      \
    } while (0)

// 64-bit key for the (tensor_id, version) offloaded map. tensor_id is a process
// pointer-derived id; version fits comfortably. We combine into a string key to
// avoid collisions across the full 128-bit space.
inline std::string tv_key(uint64_t tensor_id, uint64_t version) {
    std::string k;
    k.resize(16);
    std::memcpy(&k[0], &tensor_id, 8);
    std::memcpy(&k[8], &version, 8);
    return k;
}

}  // namespace

struct OffloadAgent::Impl {
    // ---- config / identity ----
    AgentConfig cfg;
    int device = 0;
    int numa = 0;
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;

    // ---- transport ----
    std::unique_ptr<RpcClient> rpc;

    // ---- mapped arenas (data plane) ----
    struct ArenaMap {
        uint64_t arena_id = 0;
        char* base = nullptr;
        uint64_t size = 0;
        uint32_t kind = 0;
    };
    std::vector<ArenaMap> arenas;                       // indexed order
    std::unordered_map<uint64_t, size_t> arena_index;   // arena_id -> arenas[]

    // ---- control mapping ----
    void* control_base = nullptr;
    uint64_t control_size = 0;
    ofld_slot_table_header_t* header = nullptr;
    ofld_slot_entry_t* slots = nullptr;
    uint32_t num_slots = 0;

    // ---- completion ring (shared-memory hot path) ----
    // Assigned by RegisterRank. If ring != nullptr, the progress thread pushes
    // completions into it (no RPC syscall); otherwise it uses batched RPC.
    CompletionRingHeader* comp_ring = nullptr;
    uint32_t ring_index = 0xFFFFFFFFu;

    // ---- CUDA ----
    cudaStream_t internal_stream = nullptr;

    // ---- one-off chunk registration (§6): guarded set of registered chunks ----
    std::mutex reg_mu;
    std::set<uintptr_t> registered_chunks;  // chunk start addresses
    std::atomic<uint64_t> registered_pinned_bytes{0};

    // ---- inflight bookkeeping ----
    struct InflightD2H {
        uint64_t lease_id;
        uint64_t tensor_id;
        uint64_t version;
        uint32_t slot_id;
        uint64_t nbytes;
        cudaEvent_t ev;
        bool destructive;
        InvalidateCallback invalidate_cb;
        uint64_t cookie;
        uint64_t submit_ns;
        uint64_t complete_seq;
    };
    struct InflightH2D {
        uint64_t lease_id;
        uint64_t tensor_id;
        uint64_t version;
        uint32_t slot_id;
        uint64_t nbytes;
        cudaEvent_t ev;
        bool keep_pinned;
        uint64_t submit_ns;
        uint64_t complete_seq;
    };
    std::mutex ev_mu;                 // protects the two inflight lists
    std::list<InflightD2H> inflight_d2h;
    std::list<InflightH2D> inflight_h2d;

    // ---- offloaded readiness map (tensor_id,version) -> host copy valid ----
    std::mutex off_mu;
    std::condition_variable off_cv;
    std::unordered_map<std::string, bool> offloaded;
    // Latest daemon lease per (tensor_id,version), so discard() can send a real
    // lease_id (ReleaseLeaseRequest carries no tensor identity). Written at
    // evict time; consulted by discard(). Guarded by off_mu.
    std::unordered_map<std::string, uint64_t> lease_of;
    // Ring index at which each (tensor_id,version)'s D2H completion was pushed,
    // so restore() can wait for the daemon to consume it before prefetch.
    // Guarded by off_mu.
    std::unordered_map<std::string, uint64_t> ring_push_seq;

    // ---- counters (atomic so summary() stays lock-free) ----
    std::atomic<uint64_t> inflight_d2h_bytes{0};
    std::atomic<uint64_t> inflight_h2d_bytes{0};
    std::atomic<uint64_t> evict_count{0};
    std::atomic<uint64_t> restore_count{0};
    std::atomic<uint64_t> d2h_completed{0};
    std::atomic<uint64_t> h2d_completed{0};
    std::atomic<uint64_t> submit_seq_ctr{0};
    std::atomic<uint64_t> complete_seq_ctr{0};

    // ---- background threads ----
    std::atomic<bool> stop{false};
    std::thread progress_thr;
    std::thread heartbeat_thr;

    // ==================================================================
    Impl(const AgentConfig& c) : cfg(c) {
        device = cfg.cuda_device;
        rank_id = cfg.rank_id;

        CUDA_CHECK(cudaSetDevice(device));

        // Resolve NUMA node from GPU affinity if not pinned.
        if (cfg.numa_node < 0) {
            int n = gpu_numa_node(device);
            numa = (n >= 0) ? n : 0;
        } else {
            numa = cfg.numa_node;
        }

        // Connect + register with the daemon; receive arena/control fds.
        int sock = uds_connect(cfg.socket_path);
        rpc.reset(new RpcClient(sock));

        RegisterRankRequest req;
        req.rank_id = cfg.rank_id;
        req.local_rank = cfg.local_rank;
        req.world_rank = cfg.world_rank;
        req.gpu_id = static_cast<uint32_t>(device);
        req.numa_node = static_cast<uint32_t>(numa);
        req.pid = static_cast<uint64_t>(::getpid());
        req.capabilities = 0;

        std::vector<int> fds;
        RegisterRankResponse resp = rpc->register_rank(req, &fds);
        if (!resp.ok) {
            for (int fd : fds) close_fd(fd);
            throw std::runtime_error("RegisterRank rejected: " + resp.message);
        }
        rank_epoch = resp.rank_epoch;

        // mmap each data arena at its own fd, offset 0. arena_offset in later
        // responses is relative to this per-arena base.
        try {
            for (const ArenaFd& a : resp.arenas) {
                if (a.fd_index >= fds.size())
                    throw std::runtime_error("arena fd_index out of range");
                int fd = fds[a.fd_index];
                void* base = ::mmap(nullptr, a.size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
                if (base == MAP_FAILED)
                    throw std::runtime_error(std::string("mmap arena failed: ") +
                                             std::strerror(errno));
                ArenaMap m;
                m.arena_id = a.arena_id;
                m.base = static_cast<char*>(base);
                m.size = a.size;
                m.kind = a.kind;
                arena_index[a.arena_id] = arenas.size();
                arenas.push_back(m);
            }

            // mmap the control/slot-table region.
            if (resp.control_fd_index >= fds.size())
                throw std::runtime_error("control fd_index out of range");
            control_size = resp.control_size;
            void* cbase = ::mmap(nullptr, control_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fds[resp.control_fd_index], 0);
            if (cbase == MAP_FAILED)
                throw std::runtime_error(std::string("mmap control failed: ") +
                                         std::strerror(errno));
            control_base = cbase;
        } catch (...) {
            for (int fd : fds) close_fd(fd);
            cleanup_mappings();
            throw;
        }

        // The mappings keep the fds alive; close the descriptors now.
        for (int fd : fds) close_fd(fd);

        // Validate the control header and wire up typed accessors.
        header = control_header(control_base);
        if (header->magic != OFLD_MAGIC) {
            cleanup_mappings();
            throw std::runtime_error("control header magic mismatch");
        }
        if (header->abi_version != OFLD_ABI_VERSION) {
            cleanup_mappings();
            throw std::runtime_error("control header ABI version mismatch");
        }
        slots = control_slots(control_base, header);
        num_slots = header->num_slots;

        // Locate our completion ring, if the daemon assigned one. When present,
        // the progress thread publishes completions into it (no RPC syscall);
        // otherwise it falls back to batched RPC. Validate the ring belongs to
        // us at the current epoch before trusting it.
        ring_index = resp.ring_index;
        if (ring_index != 0xFFFFFFFFu) {
            CompletionRingHeader* rh =
                ring_for_index(control_base, header, ring_index);
            if (rh && rh->owner_rank == cfg.rank_id && rh->rank_epoch == rank_epoch) {
                comp_ring = rh;
                OFLD_INFO(kTag, "using completion ring %u (cap=%u)", ring_index,
                          rh->capacity);
            } else {
                OFLD_WARN(kTag, "ring %u not owned by us; using RPC completions",
                          ring_index);
            }
        }

        // Internal (default) offload stream, used when StreamHandle==0.
        CUDA_CHECK(cudaStreamCreateWithFlags(&internal_stream, cudaStreamNonBlocking));

        // Optional eager whole-arena registration (§6). Otherwise lazy.
        if (cfg.eager_register) {
            for (const ArenaMap& m : arenas) {
                if (m.kind == OFLD_ARENA_CONTROL) continue;
                ensure_registered(reinterpret_cast<uint64_t>(m.base), m.size);
            }
        }

        // Start background threads.
        progress_thr = std::thread([this] { progress_loop(); });
        heartbeat_thr = std::thread([this] { heartbeat_loop(); });

        OFLD_INFO(kTag,
                  "rank %u registered: epoch=%llu gpu=%d numa=%d arenas=%zu slots=%u",
                  rank_id, (unsigned long long)rank_epoch, device, numa,
                  arenas.size(), num_slots);
    }

    ~Impl() {
        stop.store(true, std::memory_order_release);
        if (progress_thr.joinable()) progress_thr.join();
        if (heartbeat_thr.joinable()) heartbeat_thr.join();

        // Best-effort: drain any still-inflight events synchronously so we do
        // not destroy the stream out from under a live copy.
        drain_inflight_on_shutdown();

        if (internal_stream) {
            cudaStreamDestroy(internal_stream);  // best-effort at teardown
            internal_stream = nullptr;
        }
        cleanup_mappings();
        // rpc destructor closes the socket.
    }

    // -------------------------------------------------------------------
    void cleanup_mappings() {
        // Unregister any pinned arena ranges before unmapping, so CUDA does not
        // retain a registration over an address we are about to munmap (which
        // would fail a later mmap/register at the same address with
        // "resource already mapped").
        {
            std::lock_guard<std::mutex> lg(reg_mu);
            for (const ArenaMap& m : arenas) {
                uintptr_t abase = reinterpret_cast<uintptr_t>(m.base);
                if (m.base && registered_chunks.count(abase)) {
                    cudaHostUnregister(m.base);
                    cudaGetLastError();  // ignore unregister errors at teardown
                }
            }
            registered_chunks.clear();
        }
        for (const ArenaMap& m : arenas) {
            if (m.base) ::munmap(m.base, m.size);
        }
        arenas.clear();
        arena_index.clear();
        if (control_base) {
            ::munmap(control_base, control_size);
            control_base = nullptr;
        }
    }

    // Compute a host pointer's owning arena; returns nullptr map entry if none.
    const ArenaMap* arena_for(uintptr_t ptr) const {
        for (const ArenaMap& m : arenas) {
            uintptr_t b = reinterpret_cast<uintptr_t>(m.base);
            if (ptr >= b && ptr < b + m.size) return &m;
        }
        return nullptr;
    }

    // Ensure the arena owning [ptr, ptr+nbytes) is pinned via cudaHostRegister.
    //
    // CRITICAL: a single cudaMemcpyAsync target must lie entirely within ONE
    // registered range — CUDA rejects (invalid argument) a copy that spans two
    // separately-registered regions. Since a tensor can be larger than any
    // fixed chunk and can start at an arbitrary slot offset, we register the
    // WHOLE arena as one contiguous range on first touch (one-off per process
    // per arena, never per copy — satisfies DESIGN_SPEC §6). Arenas are
    // fd-backed host memory up to hundreds of GB; a single registration of the
    // entire mapping is exactly how large pinned arenas are meant to be pinned.
    void ensure_registered(uint64_t ptr, uint64_t nbytes) {
        if (nbytes == 0) return;
        const ArenaMap* m = arena_for(ptr);
        if (!m) {
            throw std::runtime_error("ensure_registered: pointer not in any mapped arena");
        }
        uintptr_t abase = reinterpret_cast<uintptr_t>(m->base);
        std::lock_guard<std::mutex> lg(reg_mu);
        if (registered_chunks.count(abase)) return;  // arena already registered
        cudaError_t e = cudaHostRegister(m->base, m->size, cudaHostRegisterDefault);
        if (e != cudaSuccess) {
            if (e == cudaErrorHostMemoryAlreadyRegistered) {
                cudaGetLastError();  // clear sticky error; treat as registered
            } else {
                throw std::runtime_error(std::string("cudaHostRegister(arena) failed: ") +
                                         cudaGetErrorString(e));
            }
        }
        registered_chunks.insert(abase);
        registered_pinned_bytes.fetch_add(m->size, std::memory_order_relaxed);
        OFLD_DEBUG(kTag, "registered arena %llu (%llu bytes) as one range",
                   (unsigned long long)m->arena_id, (unsigned long long)m->size);
    }

    void* slot_host_ptr(uint64_t arena_id, uint64_t arena_offset) {
        auto it = arena_index.find(arena_id);
        if (it == arena_index.end())
            throw std::runtime_error("slot references unknown arena_id");
        return arenas[it->second].base + arena_offset;
    }

    // ---- completion routines (run OUTSIDE ev_mu) --------------------------
    //
    // Each completion has two parts: (1) the RPC to the daemon updating slot
    // state, and (2) local finalize (invalidate callback, event destroy,
    // counters, readiness). The progress thread batches part (1) across all
    // completions gathered in one poll into a single BatchComplete RPC, then
    // runs part (2) per item. The synchronous wait=true paths use the single
    // *_rpc + *_finalize helpers directly.

    // Build a BatchCompletion describing a D2H completion.
    BatchCompletion make_d2h_completion(const InflightD2H& r) const {
        BatchCompletion c;
        c.event_type = static_cast<uint32_t>(EventType::kD2HComplete);
        c.lease_id = r.lease_id;
        c.tensor_id = r.tensor_id;
        c.version = r.version;
        c.slot_id = r.slot_id;
        c.seq = r.complete_seq;
        c.timestamp_ns = now_real_ns();
        c.keep_pinned = false;
        return c;
    }
    BatchCompletion make_h2d_completion(const InflightH2D& r) const {
        BatchCompletion c;
        c.event_type = static_cast<uint32_t>(EventType::kH2DComplete);
        c.lease_id = r.lease_id;
        c.tensor_id = r.tensor_id;
        c.version = r.version;
        c.slot_id = r.slot_id;
        c.seq = r.complete_seq;
        c.timestamp_ns = now_real_ns();
        c.keep_pinned = r.keep_pinned;
        return c;
    }

    // Local finalize for a D2H completion (no RPC): invalidate + counters.
    void finalize_d2h(const InflightD2H& r) {
        if (r.destructive && r.invalidate_cb) r.invalidate_cb(r.cookie);
        cudaEventDestroy(r.ev);
        {
            std::lock_guard<std::mutex> lg(off_mu);
            offloaded[tv_key(r.tensor_id, r.version)] = true;
        }
        off_cv.notify_all();
        inflight_d2h_bytes.fetch_sub(r.nbytes, std::memory_order_relaxed);
        d2h_completed.fetch_add(1, std::memory_order_relaxed);
        Metrics::instance().add(Metric::kD2HBytes, r.nbytes);
        record_bw(r.nbytes, r.submit_ns);
    }
    void finalize_h2d(const InflightH2D& r) {
        cudaEventDestroy(r.ev);
        inflight_h2d_bytes.fetch_sub(r.nbytes, std::memory_order_relaxed);
        h2d_completed.fetch_add(1, std::memory_order_relaxed);
        Metrics::instance().add(Metric::kH2DBytes, r.nbytes);
        record_bw(r.nbytes, r.submit_ns);
        off_cv.notify_all();
    }

    // Push a completion into the shared-memory ring. Returns false if the ring
    // is unavailable or full (caller must fall back to RPC). On success for a
    // D2H, records the ring index at which the entry was placed so a subsequent
    // restore of the same tensor can wait for the daemon to consume it (the
    // daemon applies ring entries asynchronously; a restore must not race ahead
    // of the location-table update).
    bool ring_push_d2h(const InflightD2H& r) {
        if (!comp_ring) return false;
        uint64_t tail_before = comp_ring->tail.load(std::memory_order_relaxed);
        ofld_completion_entry_t e;
        e.lease_id = r.lease_id;
        e.tensor_id = r.tensor_id;
        e.version = r.version;
        e.slot_id = r.slot_id;
        e.event_type = static_cast<uint32_t>(EventType::kD2HComplete);
        e.timestamp_ns = now_real_ns();
        if (!ring_push(comp_ring, e)) return false;
        {
            std::lock_guard<std::mutex> lg(off_mu);
            ring_push_seq[tv_key(r.tensor_id, r.version)] = tail_before;
        }
        return true;
    }
    bool ring_push_h2d(const InflightH2D& r) {
        if (!comp_ring) return false;
        ofld_completion_entry_t e;
        e.lease_id = r.lease_id;
        e.tensor_id = r.tensor_id;
        e.version = r.version;
        e.slot_id = r.slot_id;
        e.event_type = static_cast<uint32_t>(EventType::kH2DComplete);
        e.timestamp_ns = now_real_ns();
        return ring_push(comp_ring, e);
    }

    // Block until the daemon has consumed the ring entry at index seq (its head
    // has advanced past seq), so the daemon's location table reflects our D2H.
    void wait_ring_consumed(uint64_t seq) {
        if (!comp_ring) return;
        while (comp_ring->head.load(std::memory_order_acquire) <= seq) {
            struct timespec ts { 0, 20 * 1000L };  // 20us
            nanosleep(&ts, nullptr);
        }
    }

    // Batch the RPCs for a set of completed D2H/H2D records, then finalize each
    // locally. Used by the progress thread. Preference order for notifying the
    // daemon: (1) shared-memory ring (no syscall), (2) one BatchComplete RPC,
    // (3) single dedicated RPC. Local finalize (invalidate + counters) always
    // runs per item regardless of the notify path.
    void flush_completions_batch(const std::vector<InflightD2H>& d2h,
                                 const std::vector<InflightH2D>& h2d) {
        if (d2h.empty() && h2d.empty()) return;

        // Fast path: publish everything to the ring. Any item that doesn't fit
        // (ring full) goes to an RPC fallback so nothing is lost. Finalize each
        // item locally right after it's been handed off (ring or RPC).
        if (comp_ring) {
            BatchCompleteRequest overflow;
            overflow.rank_id = rank_id; overflow.rank_epoch = rank_epoch;
            std::vector<const InflightD2H*> of_d2h;
            std::vector<const InflightH2D*> of_h2d;
            for (const auto& r : d2h) {
                if (ring_push_d2h(r)) { finalize_d2h(r); }
                else { overflow.completions.push_back(make_d2h_completion(r)); of_d2h.push_back(&r); }
            }
            for (const auto& r : h2d) {
                if (ring_push_h2d(r)) { finalize_h2d(r); }
                else { overflow.completions.push_back(make_h2d_completion(r)); of_h2d.push_back(&r); }
            }
            if (!overflow.completions.empty()) {
                try { rpc->batch_complete(overflow); }
                catch (const std::exception& e) {
                    OFLD_ERR(kTag, "BatchComplete (ring overflow) failed: %s", e.what());
                }
                for (auto* r : of_d2h) finalize_d2h(*r);
                for (auto* r : of_h2d) finalize_h2d(*r);
            }
            return;
        }

        // No ring: single completion uses the dedicated RPC, else BatchComplete.
        if (d2h.size() + h2d.size() == 1) {
            if (!d2h.empty()) complete_d2h(d2h[0]);
            else complete_h2d(h2d[0]);
            return;
        }
        BatchCompleteRequest req;
        req.rank_id = rank_id;
        req.rank_epoch = rank_epoch;
        req.completions.reserve(d2h.size() + h2d.size());
        for (const auto& r : d2h) req.completions.push_back(make_d2h_completion(r));
        for (const auto& r : h2d) req.completions.push_back(make_h2d_completion(r));
        try {
            BatchCompleteResponse resp = rpc->batch_complete(req);
            if (!resp.ok || resp.rejected > 0) {
                OFLD_ERR(kTag, "BatchComplete: accepted=%u rejected=%u: %s",
                         resp.accepted, resp.rejected, resp.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_ERR(kTag, "BatchComplete RPC failed: %s", e.what());
        }
        // Local finalize (invalidate callbacks + counters) per item.
        for (const auto& r : d2h) finalize_d2h(r);
        for (const auto& r : h2d) finalize_h2d(r);
    }

    void complete_d2h(const InflightD2H& r) {
        // RACE_CONDITIONS §1: the CUDA event has completed, so the host copy is
        // now valid and the GPU tensor storage may be released.
        MarkD2HCompleteRequest cr;
        cr.rank_id = rank_id;
        cr.rank_epoch = rank_epoch;
        cr.lease_id = r.lease_id;
        cr.tensor_id = r.tensor_id;
        cr.version = r.version;
        cr.slot_id = r.slot_id;
        cr.complete_seq = r.complete_seq;
        cr.timestamp_ns = now_real_ns();
        try {
            MarkD2HCompleteResponse resp = rpc->mark_d2h_complete(cr);
            if (!resp.ok) {
                OFLD_ERR(kTag, "MarkD2HComplete rejected (tensor=%llu ver=%llu): %s",
                         (unsigned long long)r.tensor_id,
                         (unsigned long long)r.version, resp.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_ERR(kTag, "MarkD2HComplete RPC failed: %s", e.what());
        }
        finalize_d2h(r);
    }

    void complete_h2d(const InflightH2D& r) {
        MarkH2DCompleteRequest cr;
        cr.rank_id = rank_id;
        cr.rank_epoch = rank_epoch;
        cr.lease_id = r.lease_id;
        cr.tensor_id = r.tensor_id;
        cr.version = r.version;
        cr.slot_id = r.slot_id;
        cr.keep_pinned = r.keep_pinned;
        cr.complete_seq = r.complete_seq;
        cr.timestamp_ns = now_real_ns();
        try {
            MarkH2DCompleteResponse resp = rpc->mark_h2d_complete(cr);
            if (!resp.ok) {
                OFLD_ERR(kTag, "MarkH2DComplete rejected (tensor=%llu ver=%llu): %s",
                         (unsigned long long)r.tensor_id,
                         (unsigned long long)r.version, resp.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_ERR(kTag, "MarkH2DComplete RPC failed: %s", e.what());
        }
        finalize_h2d(r);
    }

    void record_bw(uint64_t nbytes, uint64_t submit_ns) {
        uint64_t now = now_mono_ns();
        if (now <= submit_ns) return;
        double secs = static_cast<double>(now - submit_ns) / 1e9;
        if (secs <= 0.0) return;
        double mb = static_cast<double>(nbytes) / (1024.0 * 1024.0);
        Metrics::instance().record_bandwidth_mbps(mb / secs);
    }

    // ---- progress thread --------------------------------------------------
    void progress_loop() {
        // Event queries execute against this thread's current device context.
        cudaSetDevice(device);
        pin_thread_to_node(numa);

        const long poll_us = static_cast<long>(cfg.progress_poll_us ? cfg.progress_poll_us : 200);
        while (!stop.load(std::memory_order_acquire)) {
            std::vector<InflightD2H> done_d2h;
            std::vector<InflightH2D> done_h2d;
            {
                std::lock_guard<std::mutex> lg(ev_mu);
                for (auto it = inflight_d2h.begin(); it != inflight_d2h.end();) {
                    cudaError_t st = cudaEventQuery(it->ev);
                    if (st == cudaSuccess) {
                        done_d2h.push_back(*it);
                        it = inflight_d2h.erase(it);
                    } else if (st == cudaErrorNotReady) {
                        ++it;
                    } else {
                        OFLD_ERR(kTag, "D2H event error (tensor=%llu): %s",
                                 (unsigned long long)it->tensor_id,
                                 cudaGetErrorString(st));
                        cudaEventDestroy(it->ev);
                        inflight_d2h_bytes.fetch_sub(it->nbytes, std::memory_order_relaxed);
                        it = inflight_d2h.erase(it);
                    }
                }
                for (auto it = inflight_h2d.begin(); it != inflight_h2d.end();) {
                    cudaError_t st = cudaEventQuery(it->ev);
                    if (st == cudaSuccess) {
                        done_h2d.push_back(*it);
                        it = inflight_h2d.erase(it);
                    } else if (st == cudaErrorNotReady) {
                        ++it;
                    } else {
                        OFLD_ERR(kTag, "H2D event error (tensor=%llu): %s",
                                 (unsigned long long)it->tensor_id,
                                 cudaGetErrorString(st));
                        cudaEventDestroy(it->ev);
                        inflight_h2d_bytes.fetch_sub(it->nbytes, std::memory_order_relaxed);
                        it = inflight_h2d.erase(it);
                    }
                }
            }
            // Run completions outside ev_mu. All completions gathered in this
            // poll are reported to the daemon in ONE BatchComplete RPC (falls
            // back to a single dedicated RPC when only one), then finalized
            // locally (invalidate callbacks + counters).
            flush_completions_batch(done_d2h, done_h2d);

            struct timespec ts { poll_us / 1000000L, (poll_us % 1000000L) * 1000L };
            nanosleep(&ts, nullptr);
        }
    }

    // ---- heartbeat thread -------------------------------------------------
    void heartbeat_loop() {
        const uint64_t interval_ms = cfg.heartbeat_interval_ms ? cfg.heartbeat_interval_ms : 1000;
        while (!stop.load(std::memory_order_acquire)) {
            // Sleep in small slices so shutdown is responsive.
            uint64_t slept = 0;
            while (slept < interval_ms && !stop.load(std::memory_order_acquire)) {
                struct timespec ts { 0, 10 * 1000000L };  // 10 ms
                nanosleep(&ts, nullptr);
                slept += 10;
            }
            if (stop.load(std::memory_order_acquire)) break;

            HeartbeatRequest hb;
            hb.rank_id = rank_id;
            hb.rank_epoch = rank_epoch;
            hb.timestamp_ns = now_real_ns();
            try {
                HeartbeatResponse resp = rpc->heartbeat(hb);
                if (!resp.ok) {
                    OFLD_ERR(kTag, "Heartbeat returned ok=false: %s — session likely invalidated",
                             resp.message.c_str());
                }
            } catch (const std::exception& e) {
                OFLD_ERR(kTag, "Heartbeat RPC failed: %s", e.what());
            }
        }
    }

    void drain_inflight_on_shutdown() {
        std::vector<InflightD2H> d2h;
        std::vector<InflightH2D> h2d;
        {
            std::lock_guard<std::mutex> lg(ev_mu);
            for (auto& r : inflight_d2h) d2h.push_back(r);
            for (auto& r : inflight_h2d) h2d.push_back(r);
            inflight_d2h.clear();
            inflight_h2d.clear();
        }
        for (auto& r : d2h) {
            cudaEventSynchronize(r.ev);
            complete_d2h(r);
        }
        for (auto& r : h2d) {
            cudaEventSynchronize(r.ev);
            complete_h2d(r);
        }
    }
};

// ======================================================================
// Public API
// ======================================================================

OffloadAgent::OffloadAgent(const AgentConfig& cfg) : impl_(new Impl(cfg)) {}

OffloadAgent::~OffloadAgent() = default;

OffloadTicket OffloadAgent::evict(uint64_t dev_ptr, uint64_t nbytes,
                                  StreamHandle stream, bool destructive,
                                  const TensorMeta& meta,
                                  InvalidateCallback invalidate_cb, uint64_t cookie,
                                  bool wait) {
    Impl* d = impl_.get();
    OffloadTicket t;
    t.tensor_id = meta.tensor_id;
    t.version = meta.version;
    t.nbytes = nbytes;

    cudaStream_t s = (stream == kInternalStream)
                         ? d->internal_stream
                         : reinterpret_cast<cudaStream_t>(stream);

    // Step 1: reserve a pinned slot (§7).
    RequestOffloadRequest req;
    req.rank_id = d->rank_id;
    req.rank_epoch = d->rank_epoch;
    req.tensor_id = meta.tensor_id;
    req.version = meta.version;
    req.nbytes = nbytes;
    req.gpu_id = static_cast<uint32_t>(d->device);
    req.numa_node = static_cast<uint32_t>(d->numa);
    req.alignment = 256;
    req.priority = 0;
    req.flags = OFLD_FLAG_ALLOW_OVERFLOW | (destructive ? OFLD_FLAG_DESTRUCTIVE : 0);
    req.debug_name = meta.name;

    RequestOffloadResponse resp = d->rpc->request_offload(req);
    if (!resp.ok || resp.blocked) {
        t.ok = false;
        t.message = resp.ok ? "offload blocked (pinned pressure)" : resp.message;
        return t;
    }

    d->evict_count.fetch_add(1, std::memory_order_relaxed);

    // Compute destination host pointer and ensure its chunk(s) are registered.
    void* dst = d->slot_host_ptr(resp.arena_id, resp.arena_offset);
    d->ensure_registered(reinterpret_cast<uint64_t>(dst), nbytes);
    OFLD_DEBUG(kTag, "evict d2h dst=%p src=0x%llx nbytes=%llu cap=%llu arena=%llu off=%llu",
               dst, (unsigned long long)dev_ptr, (unsigned long long)nbytes,
               (unsigned long long)resp.capacity, (unsigned long long)resp.arena_id,
               (unsigned long long)resp.arena_offset);

    uint64_t submit_seq = d->submit_seq_ctr.fetch_add(1, std::memory_order_relaxed);

    // Step 2 bookkeeping: MarkD2HSubmitted (best-effort).
    {
        MarkD2HSubmittedRequest sr;
        sr.rank_id = d->rank_id;
        sr.rank_epoch = d->rank_epoch;
        sr.lease_id = resp.lease_id;
        sr.tensor_id = meta.tensor_id;
        sr.version = meta.version;
        sr.slot_id = resp.slot_id;
        sr.submit_seq = submit_seq;
        sr.timestamp_ns = now_real_ns();
        try {
            MarkD2HSubmittedResponse ssr = d->rpc->mark_d2h_submitted(sr);
            if (!ssr.ok) {
                OFLD_WARN(kTag, "MarkD2HSubmitted ok=false (proceeding): %s",
                          ssr.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_WARN(kTag, "MarkD2HSubmitted RPC failed (proceeding): %s", e.what());
        }
    }

    // Step 2: launch D2H and record its completion event.
    uint64_t submit_ns = now_mono_ns();
    CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(dev_ptr), nbytes,
                               cudaMemcpyDeviceToHost, s));
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, s));

    d->inflight_d2h_bytes.fetch_add(nbytes, std::memory_order_relaxed);
    Metrics::instance().add(Metric::kInflightD2HBytes, nbytes);

    Impl::InflightD2H rec;
    rec.lease_id = resp.lease_id;
    rec.tensor_id = meta.tensor_id;
    rec.version = meta.version;
    rec.slot_id = resp.slot_id;
    rec.nbytes = nbytes;
    rec.ev = ev;
    rec.destructive = destructive;
    rec.invalidate_cb = invalidate_cb;
    rec.cookie = cookie;
    rec.submit_ns = submit_ns;
    rec.complete_seq = d->complete_seq_ctr.fetch_add(1, std::memory_order_relaxed);

    t.ok = true;
    t.lease_id = resp.lease_id;
    t.slot_id = resp.slot_id;

    // Record the lease for this (tensor_id,version) so discard() can release it.
    {
        std::lock_guard<std::mutex> lg(d->off_mu);
        d->lease_of[tv_key(meta.tensor_id, meta.version)] = resp.lease_id;
    }

    if (wait) {
        // Synchronous path: claim this record ourselves (never handed to the
        // progress thread), synchronize the event, then run completion inline.
        CUDA_CHECK(cudaEventSynchronize(ev));
        Metrics::instance().sub(Metric::kInflightD2HBytes, nbytes);
        d->complete_d2h(rec);
    } else {
        std::lock_guard<std::mutex> lg(d->ev_mu);
        d->inflight_d2h.push_back(rec);
    }
    return t;
}

bool OffloadAgent::is_offloaded(uint64_t tensor_id, uint64_t version) const {
    Impl* d = impl_.get();
    std::lock_guard<std::mutex> lg(d->off_mu);
    auto it = d->offloaded.find(tv_key(tensor_id, version));
    return it != d->offloaded.end() && it->second;
}

void OffloadAgent::wait_offloaded(uint64_t tensor_id, uint64_t version) {
    Impl* d = impl_.get();
    std::unique_lock<std::mutex> lk(d->off_mu);
    d->off_cv.wait(lk, [&] {
        auto it = d->offloaded.find(tv_key(tensor_id, version));
        return it != d->offloaded.end() && it->second;
    });
}

RestoreResult OffloadAgent::restore(uint64_t tensor_id, uint64_t version,
                                    uint64_t dev_ptr, uint64_t nbytes,
                                    StreamHandle stream, bool wait) {
    Impl* d = impl_.get();
    RestoreResult r;

    cudaStream_t s = (stream == kInternalStream)
                         ? d->internal_stream
                         : reinterpret_cast<cudaStream_t>(stream);

    // If this tensor's D2H completion was published to the ring, the daemon
    // applies it asynchronously. Wait until the daemon has consumed that entry
    // so its location table is up to date before we RequestPrefetch — otherwise
    // the prefetch could see a stale/absent location and fail.
    if (d->comp_ring) {
        uint64_t seq = 0;
        bool have = false;
        {
            std::lock_guard<std::mutex> lg(d->off_mu);
            auto it = d->ring_push_seq.find(tv_key(tensor_id, version));
            if (it != d->ring_push_seq.end()) { seq = it->second; have = true; }
        }
        if (have) d->wait_ring_consumed(seq);
    }

    RequestPrefetchRequest pr;
    pr.rank_id = d->rank_id;
    pr.rank_epoch = d->rank_epoch;
    pr.tensor_id = tensor_id;
    pr.version = version;
    pr.nbytes = nbytes;
    pr.gpu_id = static_cast<uint32_t>(d->device);
    pr.numa_node = static_cast<uint32_t>(d->numa);
    pr.priority = 0;
    pr.flags = 0;

    RequestPrefetchResponse presp = d->rpc->request_prefetch(pr);
    if (!presp.ok) {
        r.ok = false;
        r.message = presp.message;
        return r;
    }

    // RACE_CONDITIONS §10: must NOT H2D until the source slot is PINNED_VALID.
    // Poll (re-issuing prefetch) until ready, or time out.
    const uint64_t timeout_ns = 600ull * 1000000000ull;  // 10 min guard
    uint64_t start = now_mono_ns();
    while (!presp.ready) {
        if (now_mono_ns() - start > timeout_ns) {
            r.ok = false;
            r.message = "restore: prefetch readiness timed out";
            return r;
        }
        struct timespec ts { 0, 2 * 1000000L };  // 2 ms
        nanosleep(&ts, nullptr);
        presp = d->rpc->request_prefetch(pr);
        if (!presp.ok) {
            r.ok = false;
            r.message = presp.message;
            return r;
        }
    }

    // Defensive: confirm the resolved slot really is PINNED_VALID in shared
    // control memory before we read from it (never read a READBACK_IN_FLIGHT).
    if (presp.slot_id < d->num_slots) {
        uint32_t st = d->slots[presp.slot_id].state.load(std::memory_order_acquire);
        if (st != OFLD_SLOT_PINNED_VALID) {
            r.ok = false;
            r.message = "restore: slot not PINNED_VALID at H2D time";
            return r;
        }
    }

    d->restore_count.fetch_add(1, std::memory_order_relaxed);

    uint64_t copy_bytes = presp.nbytes ? presp.nbytes : nbytes;
    void* src = d->slot_host_ptr(presp.arena_id, presp.arena_offset);
    d->ensure_registered(reinterpret_cast<uint64_t>(src), copy_bytes);

    uint64_t resolved_version = presp.version ? presp.version : version;
    uint64_t submit_seq = d->submit_seq_ctr.fetch_add(1, std::memory_order_relaxed);

    // MarkH2DSubmitted (best-effort bookkeeping).
    {
        MarkH2DSubmittedRequest sr;
        sr.rank_id = d->rank_id;
        sr.rank_epoch = d->rank_epoch;
        sr.lease_id = presp.lease_id;
        sr.tensor_id = tensor_id;
        sr.version = resolved_version;
        sr.slot_id = presp.slot_id;
        sr.submit_seq = submit_seq;
        sr.timestamp_ns = now_real_ns();
        try {
            MarkH2DSubmittedResponse ssr = d->rpc->mark_h2d_submitted(sr);
            if (!ssr.ok) {
                OFLD_WARN(kTag, "MarkH2DSubmitted ok=false (proceeding): %s",
                          ssr.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_WARN(kTag, "MarkH2DSubmitted RPC failed (proceeding): %s", e.what());
        }
    }

    uint64_t submit_ns = now_mono_ns();
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dev_ptr), src, copy_bytes,
                               cudaMemcpyHostToDevice, s));
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, s));

    d->inflight_h2d_bytes.fetch_add(copy_bytes, std::memory_order_relaxed);
    Metrics::instance().add(Metric::kInflightH2DBytes, copy_bytes);

    Impl::InflightH2D rec;
    rec.lease_id = presp.lease_id;
    rec.tensor_id = tensor_id;
    rec.version = resolved_version;
    rec.slot_id = presp.slot_id;
    rec.nbytes = copy_bytes;
    rec.ev = ev;
    rec.keep_pinned = false;
    rec.submit_ns = submit_ns;
    rec.complete_seq = d->complete_seq_ctr.fetch_add(1, std::memory_order_relaxed);

    if (wait) {
        CUDA_CHECK(cudaEventSynchronize(ev));
        Metrics::instance().sub(Metric::kInflightH2DBytes, copy_bytes);
        d->complete_h2d(rec);
    } else {
        std::lock_guard<std::mutex> lg(d->ev_mu);
        d->inflight_h2d.push_back(rec);
    }

    r.ok = true;
    return r;
}

std::string OffloadAgent::summary_string() const {
    AgentSummary s = summary();
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "OffloadAgent rank=%u gpu=%d numa=%d epoch=%llu\n"
        "  registered_pinned_bytes=%llu\n"
        "  inflight: d2h=%llu bytes  h2d=%llu bytes\n"
        "  counts: evict=%llu restore=%llu d2h_done=%llu h2d_done=%llu\n"
        "  slots=%llu",
        s.rank_id, s.gpu_id, s.numa_node, (unsigned long long)s.rank_epoch,
        (unsigned long long)s.registered_pinned_bytes,
        (unsigned long long)s.inflight_d2h_bytes,
        (unsigned long long)s.inflight_h2d_bytes,
        (unsigned long long)s.evict_count, (unsigned long long)s.restore_count,
        (unsigned long long)s.d2h_completed, (unsigned long long)s.h2d_completed,
        (unsigned long long)s.num_slots);
    return std::string(buf);
}

AgentSummary OffloadAgent::summary() const {
    Impl* d = impl_.get();
    AgentSummary s;
    s.rank_id = d->rank_id;
    s.gpu_id = d->device;
    s.numa_node = d->numa;
    s.registered_pinned_bytes = d->registered_pinned_bytes.load(std::memory_order_relaxed);
    s.inflight_d2h_bytes = d->inflight_d2h_bytes.load(std::memory_order_relaxed);
    s.inflight_h2d_bytes = d->inflight_h2d_bytes.load(std::memory_order_relaxed);
    s.evict_count = d->evict_count.load(std::memory_order_relaxed);
    s.restore_count = d->restore_count.load(std::memory_order_relaxed);
    s.d2h_completed = d->d2h_completed.load(std::memory_order_relaxed);
    s.h2d_completed = d->h2d_completed.load(std::memory_order_relaxed);
    s.rank_epoch = d->rank_epoch;
    s.num_slots = d->num_slots;
    return s;
}

void OffloadAgent::assert_no_inflight() const {
    Impl* d = impl_.get();
    std::lock_guard<std::mutex> lg(d->ev_mu);
    if (!d->inflight_d2h.empty() || !d->inflight_h2d.empty()) {
        throw std::runtime_error("assert_no_inflight: transfers still in flight (d2h=" +
                                 std::to_string(d->inflight_d2h.size()) + " h2d=" +
                                 std::to_string(d->inflight_h2d.size()) + ")");
    }
}

std::string OffloadAgent::location(uint64_t tensor_id, uint64_t version) const {
    Impl* d = impl_.get();
    LocationQueryRequest q;
    q.tensor_id = tensor_id;
    q.version = version;
    LocationQueryResponse resp = d->rpc->query_location(q);
    if (!resp.ok) return "none";
    switch (resp.location_kind) {
        case OFLD_LOC_GPU:      return "gpu";
        case OFLD_LOC_PINNED:   return "pinned";
        case OFLD_LOC_PAGEABLE: return "pageable";
        case OFLD_LOC_NVME:     return "nvme";
        case OFLD_LOC_NONE:     return "none";
        case OFLD_LOC_DELETED:  return "none";
        case OFLD_LOC_ERROR:    return "none";
        default:                return "none";
    }
}

void OffloadAgent::discard(uint64_t tensor_id, uint64_t version) {
    Impl* d = impl_.get();
    // Resolve the lease recorded at evict time. ReleaseLeaseRequest carries no
    // tensor identity, so we must supply the concrete lease_id/slot_id here.
    uint64_t lease_id = 0;
    {
        std::lock_guard<std::mutex> lg(d->off_mu);
        auto it = d->lease_of.find(tv_key(tensor_id, version));
        if (it != d->lease_of.end()) lease_id = it->second;
    }
    if (lease_id == 0) {
        OFLD_WARN(kTag, "discard(%llu:%llu): no known lease (already released?)",
                  (unsigned long long)tensor_id, (unsigned long long)version);
    } else {
        ReleaseLeaseRequest rr;
        rr.rank_id = d->rank_id;
        rr.rank_epoch = d->rank_epoch;
        rr.lease_id = lease_id;
        rr.slot_id = 0;   // daemon resolves slot(s) from the lease record
        rr.reason = "discard tensor " + std::to_string(tensor_id) + ":" +
                    std::to_string(version);
        try {
            ReleaseLeaseResponse resp = d->rpc->release_lease(rr);
            if (!resp.ok) {
                OFLD_WARN(kTag, "discard(%llu:%llu) release_lease ok=false: %s",
                          (unsigned long long)tensor_id,
                          (unsigned long long)version, resp.message.c_str());
            }
        } catch (const std::exception& e) {
            OFLD_WARN(kTag, "discard release_lease RPC failed: %s", e.what());
        }
    }
    // Forget local readiness + lease for this identity.
    std::lock_guard<std::mutex> lg(d->off_mu);
    d->offloaded.erase(tv_key(tensor_id, version));
    d->lease_of.erase(tv_key(tensor_id, version));
}

int OffloadAgent::cuda_device() const { return impl_->device; }
int OffloadAgent::numa_node() const { return impl_->numa; }
uint32_t OffloadAgent::rank_id() const { return impl_->rank_id; }

// ---------------------------------------------------------------------------
// v2 canonical model-state API
// ---------------------------------------------------------------------------
RegisterJobResponse OffloadAgent::register_job(const RegisterJobRequest& req) {
    return impl_->rpc->register_job(req);
}

CanonicalEvictResult OffloadAgent::canonical_evict(
    uint64_t dev_ptr, uint64_t nbytes, StreamHandle stream, bool destructive,
    const RequestCanonicalEvictRequest& in_req, InvalidateCallback invalidate_cb,
    uint64_t cookie, bool hash_on_commit, bool wait) {
    Impl* d = impl_.get();
    CanonicalEvictResult out;

    cudaStream_t s = (stream == kInternalStream)
                         ? d->internal_stream
                         : reinterpret_cast<cudaStream_t>(stream);

    // Fill in rank identity + placement on the request.
    RequestCanonicalEvictRequest req = in_req;
    req.rank_id = d->rank_id;
    req.rank_epoch = d->rank_epoch;
    req.gpu_id = static_cast<uint32_t>(d->device);
    req.numa_node = static_cast<uint32_t>(d->numa);
    req.destructive = destructive;
    req.key.nbytes = nbytes;              // authoritative byte count
    if (req.key.shard_nbytes == 0) req.key.shard_nbytes = nbytes;

    RequestCanonicalEvictResponse resp = d->rpc->request_canonical_evict(req);
    if (!resp.ok) {
        out.ok = false; out.action = resp.action; out.message = resp.message;
        return out;
    }
    out.action = resp.action;
    out.object_id = resp.object_id;

    // Attach paths perform NO D2H. The caller may invalidate its local tensor
    // at a safe point (the bytes already live in the canonical object).
    if (resp.action == ATTACH_ACTION_ATTACHED_EXISTING ||
        resp.action == ATTACH_ACTION_WAIT_FOR_CREATOR) {
        out.ok = true;
        out.did_d2h = false;
        out.message = resp.message;
        // For ATTACHED_EXISTING with a destructive request, drop the local
        // storage now (safe: canonical bytes are valid). WAIT leaves the tensor
        // intact so the caller can retry.
        if (resp.action == ATTACH_ACTION_ATTACHED_EXISTING && destructive &&
            invalidate_cb) {
            invalidate_cb(cookie);
        }
        return out;
    }
    if (resp.action != ATTACH_ACTION_NEED_D2H_CREATE &&
        resp.action != ATTACH_ACTION_DUPLICATE_CANDIDATE) {
        out.ok = false; out.message = resp.message;   // quota/stale/error
        return out;
    }

    // NEED_D2H_CREATE or DUPLICATE_CANDIDATE: perform the GPU->pinned D2H into
    // the reserved slot, then commit. Reuses the same event machinery as evict.
    const uint64_t syn_tid = resp.synthetic_tid;
    void* dst = d->slot_host_ptr(resp.arena_id, resp.arena_offset);
    d->ensure_registered(reinterpret_cast<uint64_t>(dst), nbytes);
    d->evict_count.fetch_add(1, std::memory_order_relaxed);

    uint64_t submit_seq = d->submit_seq_ctr.fetch_add(1, std::memory_order_relaxed);
    {
        MarkD2HSubmittedRequest sr;
        sr.rank_id = d->rank_id; sr.rank_epoch = d->rank_epoch;
        sr.lease_id = resp.lease_id; sr.tensor_id = syn_tid;
        sr.version = 1; sr.slot_id = resp.slot_id;
        sr.submit_seq = submit_seq; sr.timestamp_ns = now_real_ns();
        try { d->rpc->mark_d2h_submitted(sr); }
        catch (const std::exception& e) {
            OFLD_WARN(kTag, "canonical MarkD2HSubmitted failed (proceeding): %s", e.what());
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<const void*>(dev_ptr), nbytes,
                               cudaMemcpyDeviceToHost, s));
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev, s));
    // Canonical create is synchronous w.r.t. the commit: we must wait for the
    // D2H so the daemon sees PINNED_VALID before CommitCanonicalObject, and so
    // we can hash the staged bytes if requested. (The VRAM-release critical
    // path is the D2H itself; the commit RPC is metadata-only.)
    CUDA_CHECK(cudaEventSynchronize(ev));
    CUDA_CHECK(cudaEventDestroy(ev));

    // MarkD2HComplete on the synthetic tid -> slot becomes PINNED_VALID and the
    // daemon publishes locations_[syn_tid].
    {
        MarkD2HCompleteRequest cr;
        cr.rank_id = d->rank_id; cr.rank_epoch = d->rank_epoch;
        cr.lease_id = resp.lease_id; cr.tensor_id = syn_tid;
        cr.version = 1; cr.slot_id = resp.slot_id;
        cr.complete_seq = d->complete_seq_ctr.fetch_add(1, std::memory_order_relaxed);
        cr.timestamp_ns = now_real_ns();
        try {
            MarkD2HCompleteResponse dr = d->rpc->mark_d2h_complete(cr);
            if (!dr.ok) OFLD_WARN(kTag, "canonical MarkD2HComplete ok=false: %s",
                                  dr.message.c_str());
        } catch (const std::exception& e) {
            out.ok = false; out.message = std::string("d2h complete rpc: ") + e.what();
            return out;
        }
    }

    uint64_t hlo = 0, hhi = 0;
    if (hash_on_commit) {
        // Hash the staged host bytes (post-D2H) for hash-verified dedup.
        hash_host_bytes(dst, nbytes, &hlo, &hhi);
    }

    CommitCanonicalObjectRequest cm;
    cm.rank_id = d->rank_id; cm.rank_epoch = d->rank_epoch;
    cm.object_id = resp.object_id; cm.lease_id = resp.lease_id;
    cm.slot_id = resp.slot_id; cm.content_hash_lo = hlo; cm.content_hash_hi = hhi;
    CommitCanonicalObjectResponse cres;
    try { cres = d->rpc->commit_canonical_object(cm); }
    catch (const std::exception& e) {
        out.ok = false; out.message = std::string("commit rpc: ") + e.what();
        return out;
    }
    out.ok = cres.ok;
    out.did_d2h = true;
    out.message = cres.message;
    // Destructive: drop local storage now that bytes are safely canonical.
    // (Losers of a duplicate-candidate race also invalidate: their bytes are
    // equivalent to the winner's; the local tensor is no longer needed.)
    if (destructive && invalidate_cb) invalidate_cb(cookie);
    return out;
}

SealModelVersionResponse OffloadAgent::seal_model_version(
    const SealModelVersionRequest& r) { return impl_->rpc->seal_model_version(r); }
GetLatestSealedVersionResponse OffloadAgent::get_latest_sealed_version(
    const GetLatestSealedVersionRequest& r) {
    return impl_->rpc->get_latest_sealed_version(r);
}
GetManifestResponse OffloadAgent::get_manifest(const GetManifestRequest& r) {
    return impl_->rpc->get_manifest(r);
}
PullTensorResponse OffloadAgent::pull_tensor(const PullTensorRequest& r) {
    return impl_->rpc->pull_tensor(r);
}

}  // namespace offload
