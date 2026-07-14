// Control-plane wire protocol for the v2 canonical model-state extension.
//
// This mirrors rpc/offload_canonical.proto field-for-field, in the same
// hand-rolled, fixed-layout binary style as protocol.h. It is ADDITIVE: v1
// framing (wire.h: [magic][opcode][flags][len][body]) and v1 messages are
// untouched. The daemon dispatch switch simply gains the v2 opcodes.
//
// Identity stack (see docs/ARCHITECTURE.md):
//   tenant_id -> job_id + launch_epoch -> model_role -> model_version
//     -> canonical tensor key
//
// The daemon owns both ends of this socket and the schema is fixed, so a
// compact little-endian codec is both safe and faster than linking protobuf
// (which would risk ODR clashes with libtorch's bundled protobuf in the rank
// process; see protocol.h).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace offload {

// ---------------------------------------------------------------------------
// Job identity. job_id + launch_epoch is identity-critical; job_name is
// human-readable metadata only (docs/design/multi-job-and-quotas.md).
// ---------------------------------------------------------------------------
struct JobKeyWire {
    uint64_t tenant_id = 0;
    uint64_t job_id = 0;
    uint64_t launch_epoch = 0;
    std::string job_name;
};

struct RegisterJobRequest {
    uint64_t tenant_id = 0;
    std::string job_name;
    // Scheduler-provided stable id (e.g. SLURM job id). 0 => daemon assigns a
    // content-hashed job_id from (tenant_id, job_name). launch_epoch is always
    // assigned fresh by the daemon so a relaunch gets a distinct namespace.
    uint64_t scheduler_job_id = 0;
    // Optional labels (flattened key/value pairs, parallel arrays).
    std::vector<std::string> label_keys;
    std::vector<std::string> label_values;
};

struct RegisterJobResponse {
    bool ok = false;
    std::string message;
    JobKeyWire job;
    uint64_t control_generation = 0;
};

// ---------------------------------------------------------------------------
// Canonical tensor key. The identity must include every axis that affects the
// bytes (partition axes) and exclude axes that merely replicate them (dp_rank).
// ---------------------------------------------------------------------------
struct CanonicalTensorKeyWire {
    JobKeyWire job;

    uint32_t model_role = 0;          // ModelRole
    uint64_t model_version = 0;

    uint64_t param_id = 0;
    uint64_t param_fqn_hash = 0;
    std::string param_fqn_debug;      // human-readable, not identity

    uint32_t dtype = 0;
    uint32_t layout = 0;
    uint64_t nbytes = 0;
    uint64_t shape_hash = 0;
    uint64_t stride_hash = 0;

    // Partition coordinates (identity-critical).
    uint32_t pp_rank = 0;
    uint32_t tp_rank = 0;
    uint32_t ep_rank = 0;
    uint32_t etp_rank = 0;
    int32_t  expert_id = -1;

    uint64_t shard_offset = 0;
    uint64_t shard_nbytes = 0;
};

struct RequestCanonicalEvictRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;

    CanonicalTensorKeyWire key;

    uint64_t local_tensor_id = 0;
    uint64_t local_version = 0;

    bool destructive = true;
    bool attach_if_exists = true;
    bool create_if_missing = true;
    uint32_t dedup_mode = 0;          // DedupMode

    // If true and another rank is CREATING this object, the daemon may hand
    // this rank a temporary candidate slot instead of making it wait.
    bool allow_duplicate_candidate = false;

    uint32_t alignment = 0;
    uint32_t priority = 0;
    uint32_t gpu_id = 0;
    uint32_t numa_node = 0;
};

struct RequestCanonicalEvictResponse {
    bool ok = false;
    std::string message;

    uint32_t action = 0;              // AttachCreateAction

    uint64_t object_id = 0;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;
    uint64_t arena_id = 0;
    uint64_t arena_offset = 0;
    uint64_t capacity = 0;
    // Daemon-synthesized tensor_id backing this (candidate or canonical) slot.
    // The rank passes it verbatim to MarkD2HSubmitted/Complete so the daemon's
    // lease/tensor checks match; the rank never constructs it itself.
    uint64_t synthetic_tid = 0;
};

struct CommitCanonicalObjectRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t object_id = 0;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;

    uint64_t content_hash_lo = 0;
    uint64_t content_hash_hi = 0;
};

struct CommitCanonicalObjectResponse {
    bool ok = false;
    std::string message;
    uint64_t object_id = 0;
    // True when this commit resolved a duplicate-candidate race as the WINNER;
    // losers get ok=true but won=false and should discard their candidate.
    bool won = true;
};

struct SealModelVersionRequest {
    JobKeyWire job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    bool fail_if_missing = true;
    // Optional expected param_ids that must all be present as VALID objects for
    // the seal to succeed (empty => seal whatever objects exist for the key).
    std::vector<uint64_t> expected_param_ids;
    // If true, atomically promote the latest-sealed pointer for this role.
    bool promote = false;
};

struct SealModelVersionResponse {
    bool ok = false;
    std::string message;
    uint32_t state = 0;               // ManifestState
    uint64_t tensor_count = 0;
    uint64_t total_nbytes = 0;
    uint64_t manifest_hash_lo = 0;
    uint64_t manifest_hash_hi = 0;
};

struct GetLatestSealedVersionRequest {
    JobKeyWire job;
    uint32_t model_role = 0;
};

struct GetLatestSealedVersionResponse {
    bool ok = false;
    std::string message;
    bool found = false;
    uint64_t model_version = 0;
};

// Optional rollout-side filters on GetManifest (proto ManifestFilter). A
// *_present flag of false means "unset / match all" for that field.
struct ManifestFilterWire {
    bool pp_present = false;   uint32_t pp_rank = 0;
    bool tp_present = false;   uint32_t tp_rank = 0;
    bool ep_present = false;   uint32_t ep_rank = 0;
    bool etp_present = false;  uint32_t etp_rank = 0;
    bool expert_present = false; int32_t expert_id = 0;
};

struct GetManifestRequest {
    JobKeyWire job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    ManifestFilterWire filter;
};

struct TensorManifestEntryWire {
    CanonicalTensorKeyWire key;
    uint64_t object_id = 0;
    uint64_t nbytes = 0;
    uint64_t content_hash_lo = 0;
    uint64_t content_hash_hi = 0;
    std::string location_hint;        // e.g. "node3:nvme", "pinned", "pageable"
};

struct GetManifestResponse {
    bool ok = false;
    std::string message;
    JobKeyWire job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    uint32_t state = 0;               // ManifestState
    std::vector<TensorManifestEntryWire> tensors;
};

struct PullTensorRequest {
    JobKeyWire job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    uint64_t object_id = 0;

    uint32_t transport = 0;           // TransportKind

    // Opaque transport-specific target info: for TCP this is "host:port"; for
    // FILE this is a destination path; for libfabric this is an endpoint /
    // rkey / remote-address descriptor. Kept opaque at this layer.
    std::string target_descriptor;
};

struct PullTensorResponse {
    bool ok = false;
    std::string message;
    uint64_t object_id = 0;
    uint64_t nbytes = 0;
    uint32_t transport = 0;
    uint64_t content_hash_lo = 0;
    uint64_t content_hash_hi = 0;
    // Opaque transport-specific metadata (transfer id, staging id, rkey, ...).
    std::string transport_metadata;
};

// ---------------------------------------------------------------------------
// Local canonical restore (trainer offload/reload round-trip). A rank asks the
// daemon to make a canonical object readable in a pinned arena and hand back
// the (arena_id, arena_offset) of its bytes; the rank then H2Ds directly from
// that shared pinned region into its own GPU storage. This is how a DP replica
// that got ATTACHED_EXISTING (no local D2H, no rank-local tensor_id) reads the
// shared bytes back. Strictly read-only: the daemon holds a restore refcount so
// drain/recycle cannot race the H2D, and never frees the object's backing here.
// ---------------------------------------------------------------------------
struct RequestCanonicalRestoreRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t object_id = 0;
    uint32_t gpu_id = 0;      // requester's GPU/NUMA, for a readback slot if cold
    uint32_t numa_node = 0;
};

struct RequestCanonicalRestoreResponse {
    bool ok = false;
    std::string message;
    // True when the object is momentarily transitioning between tiers (e.g. a
    // drain is in flight): the caller should retry shortly rather than fail.
    bool retriable = false;
    uint64_t object_id = 0;
    uint64_t nbytes = 0;
    // Location of the object's bytes in a pinned arena the rank has mapped. The
    // rank computes its local host pointer from (arena_id, arena_offset) and
    // H2Ds into its GPU buffer. Valid until ReleaseCanonicalRestore.
    uint64_t arena_id = 0;
    uint64_t arena_offset = 0;
};

struct ReleaseCanonicalRestoreRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t object_id = 0;
};

struct ReleaseCanonicalRestoreResponse {
    bool ok = false;
    std::string message;
};

// Explicit version GC. Releases every canonical object of (job, model_role,
// model_version) and its manifest. Honors in-flight export/restore holds
// (those objects are skipped and reported so the caller can retry); it does
// NOT honor the advisory attachment refcount — this is a coarse "the whole
// version is dead" GC the trainer issues once the version is no longer needed.
struct DropCanonicalVersionRequest {
    JobKeyWire job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
};

struct DropCanonicalVersionResponse {
    bool ok = false;
    std::string message;
    uint64_t dropped_count = 0;      // objects freed
    uint64_t skipped_inflight = 0;   // objects skipped (export/restore in flight)
    uint64_t bytes_freed = 0;        // sum of freed object nbytes
};

}  // namespace offload
