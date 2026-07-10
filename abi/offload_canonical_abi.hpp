#pragma once

#include <atomic>
#include <cstdint>

namespace offload_v2 {

// ABI version for canonical-object extensions.
static constexpr uint32_t OFFLOAD_CANONICAL_ABI_VERSION = 2;
static constexpr uint32_t OFFLOAD_CANONICAL_MAGIC = 0x4f464332; // "OFC2"

using TenantId = uint64_t;
using JobId = uint64_t;
using LaunchEpoch = uint64_t;
using ModelVersion = uint64_t;
using ParamId = uint64_t;
using ObjectId = uint64_t;
using LeaseId = uint64_t;
using SlotId = uint32_t;

// Keep values stable across ABI versions.
enum ModelRole : uint32_t {
    MODEL_ROLE_UNKNOWN          = 0,
    MODEL_ROLE_POLICY_TRAINABLE = 1,
    MODEL_ROLE_POLICY_ROLLOUT   = 2,
    MODEL_ROLE_POLICY_REFERENCE = 3,
    MODEL_ROLE_REWARD_MODEL     = 4,
    MODEL_ROLE_VALUE_MODEL      = 5,
    MODEL_ROLE_AUXILIARY        = 6,
};

enum CanonicalObjectState : uint32_t {
    OBJ_MISSING          = 0,
    OBJ_CREATING         = 1,
    OBJ_VALID_PINNED     = 2,
    OBJ_VALID_PAGEABLE   = 3,
    OBJ_VALID_NVME       = 4,
    OBJ_VALID_MULTI_TIER = 5,
    OBJ_SEALED           = 6,
    OBJ_EXPORT_IN_FLIGHT = 7,
    OBJ_DELETED          = 8,
    OBJ_ERROR            = 255,
};

enum ManifestState : uint32_t {
    MANIFEST_OPEN     = 0,
    MANIFEST_SEALING  = 1,
    MANIFEST_SEALED   = 2,
    MANIFEST_REVOKED  = 3,
    MANIFEST_ERROR    = 255,
};

enum DedupMode : uint32_t {
    DEDUP_DISABLED          = 0,
    DEDUP_SEMANTIC_TRUSTED  = 1,
    DEDUP_HASH_VERIFIED     = 2,
    DEDUP_SAMPLED_HASH      = 3,
};

enum AttachCreateAction : uint32_t {
    ATTACH_ACTION_ATTACHED_EXISTING      = 0,
    ATTACH_ACTION_NEED_D2H_CREATE        = 1,
    ATTACH_ACTION_WAIT_FOR_CREATOR       = 2,
    ATTACH_ACTION_DUPLICATE_CANDIDATE    = 3,
    ATTACH_ACTION_REJECTED_STALE_VERSION = 4,
    ATTACH_ACTION_QUOTA_EXCEEDED         = 5,
    ATTACH_ACTION_ERROR                  = 255,
};

struct alignas(64) JobRecord {
    TenantId tenant_id;
    JobId job_id;
    LaunchEpoch launch_epoch;

    // Human-readable only. Not identity-critical.
    char job_name[128];

    std::atomic<uint32_t> state;
    uint32_t priority;
    uint64_t created_ns;
    uint64_t heartbeat_ns;

    std::atomic<uint64_t> pinned_bytes;
    std::atomic<uint64_t> pageable_bytes;
    std::atomic<uint64_t> nvme_bytes;
    std::atomic<uint64_t> inflight_d2h_bytes;
    std::atomic<uint64_t> inflight_h2d_bytes;
    std::atomic<uint64_t> inflight_export_bytes;

    uint64_t max_pinned_bytes;
    uint64_t max_pageable_bytes;
    uint64_t max_nvme_bytes;
    uint64_t max_inflight_d2h_bytes;

    uint64_t reserved[8];
};

struct alignas(64) CanonicalTensorKeyAbi {
    TenantId tenant_id;
    JobId job_id;
    LaunchEpoch launch_epoch;

    uint32_t model_role;
    uint32_t dtype;
    ModelVersion model_version;

    ParamId param_id;
    uint64_t param_fqn_hash;

    uint32_t layout;
    uint32_t reserved0;
    uint64_t nbytes;
    uint64_t shape_hash;
    uint64_t stride_hash;

    uint32_t pp_rank;
    uint32_t tp_rank;
    uint32_t ep_rank;
    uint32_t etp_rank;
    int32_t expert_id;
    uint32_t reserved1;

    uint64_t shard_offset;
    uint64_t shard_nbytes;

    uint64_t reserved[8];
};

struct alignas(64) CanonicalObjectEntry {
    ObjectId global_object_id;
    CanonicalTensorKeyAbi key;

    std::atomic<uint32_t> state;
    std::atomic<uint32_t> refcount;
    std::atomic<uint32_t> export_refcount;
    uint32_t producer_rank;

    uint64_t nbytes;
    uint64_t content_hash_lo;
    uint64_t content_hash_hi;

    SlotId pinned_slot;
    uint32_t flags;
    uint64_t pageable_ref;
    uint64_t nvme_ref;

    uint64_t created_ns;
    uint64_t sealed_ns;
    uint64_t last_access_ns;

    uint64_t reserved[8];
};

struct alignas(64) ManifestHeaderEntry {
    TenantId tenant_id;
    JobId job_id;
    LaunchEpoch launch_epoch;
    uint32_t model_role;
    uint32_t state;
    ModelVersion model_version;

    uint64_t num_tensors;
    uint64_t total_nbytes;
    uint64_t created_ns;
    uint64_t sealed_ns;
    uint64_t manifest_hash_lo;
    uint64_t manifest_hash_hi;

    uint64_t reserved[8];
};

struct alignas(64) TensorManifestEntryAbi {
    ObjectId object_id;
    CanonicalTensorKeyAbi key;
    uint64_t content_hash_lo;
    uint64_t content_hash_hi;
    uint64_t location_hint_node_id;
    uint64_t location_hint_kind;
    uint64_t reserved[8];
};

struct alignas(64) CanonicalControlHeader {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t generation;

    uint64_t job_table_offset;
    uint64_t job_table_capacity;

    uint64_t object_table_offset;
    uint64_t object_table_capacity;

    uint64_t manifest_table_offset;
    uint64_t manifest_table_capacity;

    uint64_t reserved[16];
};

} // namespace offload_v2
