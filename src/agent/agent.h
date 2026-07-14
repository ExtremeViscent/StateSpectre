// Rank-side agent: public C++ API.
//
// This is the interface the Python extension (and C++ tests) bind against. The
// agent owns all CUDA interaction (streams, events, cudaHostRegister, D2H/H2D
// copies) and all daemon RPC. Callers pass raw device pointers + a stream
// handle, so this header does NOT include CUDA or torch — the boundary stays
// clean and the Python extension links the agent without CUDA symbol coupling
// leaking into pybind translation units.
//
// Threading: OffloadAgent is internally synchronized. evict()/restore() may be
// called from any thread. A background progress thread polls CUDA events and
// reports completions to the daemon; a heartbeat thread keeps the session live.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "protocol_v2.h"

namespace offload {

// Opaque CUDA stream handle (a cudaStream_t reinterpreted as uintptr_t).
//
// A value of 0 is the CUDA *default* stream (cudaStream_t 0) and is used as-is,
// so a caller passing the process's real default/current stream gets correct
// producer->D2H ordering on that stream. To ask the agent to use its own
// internal non-blocking offload stream instead, pass kInternalStream.
//
// IMPORTANT: do not overload 0 to mean "internal stream" — 0 is a real, valid
// CUDA stream, and silently substituting a different stream would break
// stream ordering between a producer kernel and the D2H copy.
using StreamHandle = uintptr_t;
static constexpr StreamHandle kInternalStream = ~static_cast<StreamHandle>(0);

// Callback invoked (from the progress thread) after a destructive eviction's
// D2H event completes, so the owner can replace the original tensor's storage
// with empty storage. cookie is whatever the caller passed to evict().
using InvalidateCallback = std::function<void(uint64_t cookie)>;

struct AgentConfig {
    std::string socket_path = "/tmp/state_spectre.sock";
    int cuda_device = 0;         // local CUDA device index this rank drives
    uint32_t rank_id = 0;
    uint32_t local_rank = 0;
    uint32_t world_rank = 0;
    int numa_node = -1;          // -1 => auto-detect from GPU affinity
    // registration is one-off per mapped chunk; this is the chunk size used
    // when lazily registering arena regions with cudaHostRegister.
    uint64_t registration_chunk_bytes = 512ull << 20;
    bool eager_register = false; // if true, register whole arena at startup
    uint64_t heartbeat_interval_ms = 1000;
    uint64_t progress_poll_us = 200;  // progress-thread event poll interval
};

// Metadata captured at evict time; also what a handle carries for restore.
struct TensorMeta {
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint64_t nbytes = 0;
    std::string name;
    // Shape/stride/dtype are opaque to the agent; the Python layer keeps them.
};

// Result of an evict: the caller holds this to later restore.
struct OffloadTicket {
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint64_t nbytes = 0;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;
    bool ok = false;
    std::string message;
};

// Result of a restore request.
struct RestoreResult {
    bool ok = false;
    std::string message;
};

// Result of a canonical evict (attach-or-create). When action ==
// NEED_D2H_CREATE the agent performed a D2H and committed the object; when
// ATTACHED_EXISTING no D2H happened (bytes deduped). See offload_canonical_abi.
struct CanonicalEvictResult {
    bool ok = false;
    std::string message;
    uint32_t action = 0;        // AttachCreateAction
    uint64_t object_id = 0;
    bool did_d2h = false;
};

// Per-rank live summary (subset of METRICS.md per-rank block).
struct AgentSummary {
    uint32_t rank_id = 0;
    int gpu_id = 0;
    int numa_node = 0;
    uint64_t registered_pinned_bytes = 0;
    uint64_t inflight_d2h_bytes = 0;
    uint64_t inflight_h2d_bytes = 0;
    uint64_t evict_count = 0;
    uint64_t restore_count = 0;
    uint64_t d2h_completed = 0;
    uint64_t h2d_completed = 0;
    uint64_t rank_epoch = 0;
    uint64_t num_slots = 0;
};

class OffloadAgent {
 public:
    // Connect to the daemon, RegisterRank, receive + mmap fds, start threads.
    // Throws std::runtime_error on failure.
    explicit OffloadAgent(const AgentConfig& cfg);
    ~OffloadAgent();

    OffloadAgent(const OffloadAgent&) = delete;
    OffloadAgent& operator=(const OffloadAgent&) = delete;

    // Destructive/ non-destructive eviction of a contiguous device buffer.
    //   dev_ptr : device pointer to the tensor's storage (byte 0)
    //   nbytes  : number of bytes to copy
    //   stream  : CUDA stream to order the D2H on (0 => internal stream)
    //   destructive : if true, invalidate_cb(cookie) fires after D2H completes
    //   invalidate_cb / cookie : storage-replacement hook (may be null)
    // Returns a ticket. The D2H is launched before returning; completion is
    // reported asynchronously by the progress thread. If wait==true, blocks
    // until the D2H event completes (and, for destructive, until invalidated).
    OffloadTicket evict(uint64_t dev_ptr, uint64_t nbytes, StreamHandle stream,
                        bool destructive, const TensorMeta& meta,
                        InvalidateCallback invalidate_cb, uint64_t cookie,
                        bool wait);

    // Query whether a previously-evicted tensor's D2H has completed
    // (host copy valid). Non-blocking.
    bool is_offloaded(uint64_t tensor_id, uint64_t version) const;

    // Block until the given tensor's D2H completes.
    void wait_offloaded(uint64_t tensor_id, uint64_t version);

    // Restore a tensor into the provided device buffer.
    //   tensor_id/version : identity (version 0 => latest)
    //   dev_ptr : destination device pointer (already allocated, nbytes big)
    //   nbytes  : bytes to copy
    //   stream  : CUDA stream to order H2D on
    //   wait    : block until H2D completes before returning
    // Handles prefetch (cold->pinned readback) transparently before H2D.
    RestoreResult restore(uint64_t tensor_id, uint64_t version, uint64_t dev_ptr,
                          uint64_t nbytes, StreamHandle stream, bool wait);

    // Human-readable multi-line summary + structured accessor.
    std::string summary_string() const;
    AgentSummary summary() const;

    // Assert nothing is in flight (throws std::runtime_error if something is).
    void assert_no_inflight() const;

    // Location string for a tensor ("gpu"/"pinned"/"pageable"/"nvme"/"none").
    std::string location(uint64_t tensor_id, uint64_t version) const;

    // Explicitly drop a tensor's daemon-side lease/metadata (best-effort).
    void discard(uint64_t tensor_id, uint64_t version);

    int cuda_device() const;
    int numa_node() const;
    uint32_t rank_id() const;

    // ---- v2 canonical model-state API ----
    // Register this session's job. Fills the returned JobKeyWire (identity) so
    // subsequent canonical keys carry it. Throws on RPC failure.
    RegisterJobResponse register_job(const RegisterJobRequest& req);

    // Canonical evict (attach-or-create). Reserves a canonical object via the
    // daemon; on NEED_D2H_CREATE performs the GPU->pinned D2H (reusing the same
    // event machinery as evict) and commits the object, optionally hashing the
    // staged bytes for hash-verified dedup. On ATTACHED_EXISTING / WAIT no copy
    // is done. `dev_ptr`/`nbytes` describe the source tensor; `invalidate_cb`
    // fires after a real D2H completes (destructive path), like evict().
    CanonicalEvictResult canonical_evict(uint64_t dev_ptr, uint64_t nbytes,
                                         StreamHandle stream, bool destructive,
                                         const RequestCanonicalEvictRequest& req,
                                         InvalidateCallback invalidate_cb,
                                         uint64_t cookie, bool hash_on_commit,
                                         bool wait);

    // Manifest / rollout control-plane passthroughs (used by trainer + rollout).
    SealModelVersionResponse       seal_model_version(const SealModelVersionRequest& r);
    GetLatestSealedVersionResponse get_latest_sealed_version(
        const GetLatestSealedVersionRequest& r);
    GetManifestResponse            get_manifest(const GetManifestRequest& r);
    PullTensorResponse             pull_tensor(const PullTensorRequest& r);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offload
