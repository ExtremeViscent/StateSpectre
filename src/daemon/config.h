// Configuration model + parser for the daemon.
//
// The config file uses the YAML subset shown in spec/config.example.yaml.
// We parse that subset directly (indentation-based key/value + simple lists of
// maps) rather than pulling in a YAML library, so the daemon has no extra
// runtime dependency. Unknown keys are ignored with a warning; missing keys
// fall back to documented defaults.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace offload {

enum class DrainTarget {
    kPageableOnly,
    kNvmeOnly,
    kPageableThenNvme,
};

struct GpuWindowConfig {
    uint32_t gpu = 0;
    uint64_t size_bytes = 0;
};

struct NumaArenaConfig {
    uint32_t numa_node = 0;
    uint64_t size_bytes = 0;                 // total pinned arena for this node
    std::vector<GpuWindowConfig> gpu_windows;
    uint64_t overflow_bytes = 0;
};

struct DaemonConfig {
    // daemon:
    std::string socket_path = "/tmp/fastoffload.sock";
    std::string control_shm_name = "/fastoffload_ctrl";
    uint64_t heartbeat_timeout_ms = 5000;

    // arenas:
    std::string registration_policy = "lazy_chunked";  // informational
    uint64_t registration_chunk_bytes = 512ull << 20;
    uint64_t allocation_granularity_bytes = 16ull << 20;
    std::vector<NumaArenaConfig> per_numa;

    // drain_policy:
    DrainTarget drain_target = DrainTarget::kPageableThenNvme;
    bool drain_on_d2h_complete = true;
    double normal_threshold = 0.60;
    double priority_threshold = 0.80;
    double aggressive_threshold = 0.95;

    // nvme:
    bool nvme_enabled = true;
    std::string nvme_path = "/nvme/fastoffload";
    std::string nvme_io_engine = "io_uring";           // io_uring | pwrite
    bool nvme_direct_io = true;
    bool nvme_stripe = true;
    // Number of stripe files a single tensor is split across. With multiple
    // physical drives, point nvme_stripe_dirs at one dir per drive so stripes
    // land on different devices. Larger blocks + higher queue depth matter a
    // lot for O_DIRECT throughput (measured: 1MB->16MB gives +20-25%).
    uint32_t nvme_stripe_count = 4;
    uint64_t nvme_block_bytes = 16ull << 20;           // IO chunk per submission
    uint32_t nvme_queue_depth = 8;                     // io_uring in-flight SQEs
    // Optional explicit per-drive directories; if non-empty, stripe i uses
    // nvme_stripe_dirs[i % size]. Otherwise all stripes live under nvme_path.
    std::vector<std::string> nvme_stripe_dirs;

    // python_api (informational; enforced rank-side):
    std::string default_invalidate = "set_empty";
    bool require_own_storage = true;
    bool allow_views = false;
    bool unsafe_autograd = false;

    // number of background drain worker threads
    uint32_t drain_workers = 2;
    // number of NVMe io worker threads (when engine == pwrite)
    uint32_t nvme_workers = 2;

    // completion rings (shared-memory hot path). max_ranks rings are reserved
    // in the control region; each holds ring_capacity entries. 0 max_ranks
    // disables rings (ranks fall back to batched RPC completions).
    uint32_t completion_ring_max_ranks = 64;
    uint32_t completion_ring_capacity = 1024;   // must be power of two
    // Interval the daemon's ring-poller sleeps between sweeps (microseconds).
    uint32_t ring_poll_us = 50;

    // -------------------------------------------------------------------
    // v2 canonical model-state store (offload_manager_v2.yaml). All additive;
    // when enable_canonical_store is false the daemon behaves exactly like v1.
    // -------------------------------------------------------------------
    bool v2_enable_canonical_store = true;
    bool v2_enable_manifests = true;
    bool v2_enable_rollout_export = true;

    // namespace:
    uint64_t v2_default_tenant_id = 0;
    bool v2_require_job_registration = true;
    bool v2_include_launch_epoch = true;

    // canonical_store:
    // dedup_default_mode: 0 disabled | 1 semantic_trusted | 2 hash_verified | 3 sampled_hash
    uint32_t v2_dedup_default_mode = 1;
    bool v2_cross_job_dedup = false;
    // creating_policy: 0 wait | 1 duplicate_candidate_on_pressure
    uint32_t v2_creating_policy = 0;
    double v2_duplicate_candidate_gpu_pressure_threshold = 0.90;
    uint64_t v2_sampled_hash_bytes_per_gib = 1ull << 20;

    // manifest:
    bool v2_seal_requires_all_objects_valid = true;
    uint32_t v2_retain_sealed_versions = 4;

    // export:
    // default_transport: 1 tcp | 2 file | 3 libfabric_send
    uint32_t v2_default_transport = 3;
    uint32_t v2_fallback_transport = 1;
    bool v2_use_dedicated_export_buffers = true;
    bool v2_require_export_refcount = true;
    uint32_t v2_max_concurrent_exports_per_job = 8;

    // Optional TCP control endpoint for rollout engines (host-order port). 0
    // disables; the daemon then serves canonical RPCs over the UDS only.
    uint32_t v2_control_tcp_port = 0;

    // default per-job quotas (0 == unlimited).
    uint64_t v2_quota_max_pinned_bytes = 0;
    uint64_t v2_quota_max_pageable_bytes = 0;
    uint64_t v2_quota_max_nvme_bytes = 0;
    uint64_t v2_quota_max_inflight_d2h_bytes = 0;
    uint64_t v2_quota_max_inflight_export_bytes = 0;

    // Total slot count across all arenas is derived from arena sizes and
    // allocation granularity at build time; see BuildSlotPlan.
};

// Parse config from a file path. On any structural error throws
// std::runtime_error with a descriptive message. Applies defaults for anything
// omitted. If path is empty, returns an all-default single-NUMA config sized
// for a smoke test (small arena) unless overridden by env.
DaemonConfig load_config(const std::string& path);

// Build a config suitable for a minimal single-GPU smoke deployment with the
// given pinned arena size. Used by tests and when no config file is present.
DaemonConfig default_smoke_config(uint64_t arena_bytes,
                                  uint32_t numa_node,
                                  uint32_t gpu_id);

const char* drain_target_name(DrainTarget t);

}  // namespace offload
