// Daemon-side data model for the v2 canonical model-state object store.
//
// These live in the daemon heap alongside the existing leases_/locations_
// tables (not in the shared control region) — the rank never reads them
// directly; it interacts through the v2 RPCs in protocol_v2.h. The canonical
// ABI in abi/offload_canonical_abi.hpp documents the intended shared-memory
// layout for a future zero-copy control region; v2 keeps the authoritative
// tables in the daemon for the same reason v1 keeps leases/locations there.
//
// Physical backing reuse: a canonical object does NOT re-implement tiering. It
// is backed by the existing v1 slot/lease/drain/cold machinery via a synthetic
// tensor_id (kCanonicalTidBit | object_id). Its current physical location is
// therefore always locations_[synthetic_tid]; drain/readback/NVMe all apply
// unchanged.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "offload_canonical_abi.hpp"
#include "protocol_v2.h"

namespace offload {

// The canonical ABI defines the object/manifest/role enums in namespace
// offload_v2 (abi/offload_canonical_abi.hpp). Bring them into scope here so the
// daemon tables can use the same stable numeric values the RPC layer carries.
using offload_v2::CanonicalObjectState;
using offload_v2::ManifestState;
using offload_v2::ModelRole;
using offload_v2::DedupMode;
using offload_v2::AttachCreateAction;
using namespace offload_v2;  // OBJ_*, MANIFEST_*, MODEL_ROLE_*, DEDUP_*, ATTACH_*

// Top bit of a tensor_id marks a daemon-synthesized canonical-object tensor.
// Rank-assigned ids are (rank<<40 | counter) and never set bit 63.
static constexpr uint64_t kCanonicalTidBit = (1ull << 63);
inline uint64_t canonical_tid(uint64_t object_id) { return kCanonicalTidBit | object_id; }
inline bool is_canonical_tid(uint64_t tid) { return (tid & kCanonicalTidBit) != 0; }

// Composite job identity. (tenant_id, job_id, launch_epoch) is the true key;
// job_name is display-only metadata.
struct JobUID {
    uint64_t tenant_id = 0;
    uint64_t job_id = 0;
    uint64_t launch_epoch = 0;
    bool operator==(const JobUID& o) const {
        return tenant_id == o.tenant_id && job_id == o.job_id &&
               launch_epoch == o.launch_epoch;
    }
};
struct JobUIDHash {
    size_t operator()(const JobUID& j) const {
        uint64_t h = j.tenant_id * 1099511628211ull;
        h = (h ^ j.job_id) * 1099511628211ull;
        h = (h ^ j.launch_epoch) * 1099511628211ull;
        return static_cast<size_t>(h);
    }
};

enum class JobState : uint32_t {
    kRegistered = 0,
    kRunning = 1,
    kDraining = 2,
    kTerminating = 3,
    kTerminated = 4,
    kError = 255,
};

struct JobRecord {
    JobUID uid;
    std::string job_name;
    JobState state = JobState::kRunning;
    uint32_t priority = 0;
    uint64_t created_ns = 0;
    uint64_t heartbeat_ns = 0;

    // Per-job resource accounting (04_MULTI_JOB_NAMESPACE_AND_QUOTAS.md).
    uint64_t pinned_bytes = 0;
    uint64_t pageable_bytes = 0;
    uint64_t nvme_bytes = 0;
    uint64_t inflight_d2h_bytes = 0;
    uint64_t inflight_export_bytes = 0;
    uint64_t canonical_object_count = 0;

    // Quotas (0 == unlimited).
    uint64_t max_pinned_bytes = 0;
    uint64_t max_pageable_bytes = 0;
    uint64_t max_nvme_bytes = 0;
    uint64_t max_inflight_d2h_bytes = 0;
    uint64_t max_inflight_export_bytes = 0;
};

// A canonical object: the immutable bytes of one logical model-state shard at
// one sealed model version. Backed physically by a synthetic-tid slot/lease.
struct CanonicalObject {
    uint64_t object_id = 0;
    CanonicalTensorKeyWire key;      // full identity (dp excluded by construction)
    std::string key_str;             // serialized identity, index key

    uint64_t nbytes = 0;
    CanonicalObjectState state = OBJ_MISSING;

    uint32_t refcount = 0;           // rank attachments
    uint32_t export_refcount = 0;    // in-flight remote transfers
    uint32_t producer_rank = 0;
    uint32_t flags = 0;

    uint64_t content_hash_lo = 0;
    uint64_t content_hash_hi = 0;

    // Physical backing (v1 machinery). synthetic_tid resolves current location
    // through locations_[synthetic_tid]; base_slot/slot_count/lease_id are the
    // creating lease (valid while pinned).
    uint64_t synthetic_tid = 0;
    uint64_t version = 1;
    uint64_t lease_id = 0;
    uint32_t base_slot = 0xFFFFFFFFu;
    uint32_t slot_count = 0;

    // Creating-race bookkeeping.
    uint32_t creating_rank = 0xFFFFFFFFu;

    uint64_t created_ns = 0;
    uint64_t sealed_ns = 0;
    uint64_t last_access_ns = 0;
};

// A sealed (or in-progress) model-version manifest.
struct Manifest {
    JobUID job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    ManifestState state = MANIFEST_OPEN;

    std::vector<uint64_t> object_ids;  // members, in insertion order
    uint64_t total_nbytes = 0;
    uint64_t created_ns = 0;
    uint64_t sealed_ns = 0;
    uint64_t manifest_hash_lo = 0;
    uint64_t manifest_hash_hi = 0;
};

// Index key for (job, role, model_version).
struct VersionKey {
    JobUID job;
    uint32_t model_role = 0;
    uint64_t model_version = 0;
    bool operator==(const VersionKey& o) const {
        return job == o.job && model_role == o.model_role &&
               model_version == o.model_version;
    }
};
struct VersionKeyHash {
    size_t operator()(const VersionKey& v) const {
        size_t h = JobUIDHash{}(v.job);
        h = (h ^ (v.model_role * 2654435761u)) * 1099511628211ull;
        h = (h ^ v.model_version) * 1099511628211ull;
        return h;
    }
};

// Index key for (job, role) -> latest promoted sealed version.
struct RoleKey {
    JobUID job;
    uint32_t model_role = 0;
    bool operator==(const RoleKey& o) const {
        return job == o.job && model_role == o.model_role;
    }
};
struct RoleKeyHash {
    size_t operator()(const RoleKey& r) const {
        return JobUIDHash{}(r.job) ^ (r.model_role * 2654435761u);
    }
};

}  // namespace offload
