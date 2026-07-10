# Canonical Model-State Object Store

## Motivation

In distributed LLM RL post-training, many ranks hold identical copies of the same model-state shard along replicated axes. Instead of storing all rank-local copies in host RAM/NVMe, the offload manager should store one canonical object and let multiple rank-local tensor records attach to it.

## Base Object Identity

A canonical object is identified by:

```text
tenant_id
job_id
launch_epoch
model_role
model_version
param_id / param_fqn_hash
parallel partition coordinates
shard range
shape/dtype/layout metadata
```

The identity must include all axes that affect tensor bytes and exclude axes that merely replicate bytes.

## Replicated vs Partitioned Axes

Usually exclude:

```text
dp_rank
replica_rank
local process id
node id
```

Usually include:

```text
pp_rank
tp_rank
ep_rank
etp_rank
expert_id
shard_offset
shard_nbytes
```

Deduplication is legal only across replicated axes. Partition axes are part of the canonical key.

## Canonical Key

```cpp
struct CanonicalTensorKey {
    uint64_t tenant_id;
    uint64_t job_id;
    uint64_t launch_epoch;

    uint32_t model_role;
    uint64_t model_version;

    uint64_t param_id;
    uint64_t param_fqn_hash;

    uint32_t dtype;
    uint32_t layout;
    uint64_t nbytes;
    uint64_t shape_hash;
    uint64_t stride_hash;

    uint32_t pp_rank;
    uint32_t tp_rank;
    uint32_t ep_rank;
    uint32_t etp_rank;
    int32_t  expert_id;

    uint64_t shard_offset;
    uint64_t shard_nbytes;
};
```

## Canonical Object

```cpp
struct CanonicalObject {
    uint64_t global_object_id;

    CanonicalTensorKey key;

    uint64_t nbytes;
    uint32_t state;

    uint32_t refcount;
    uint32_t export_refcount;
    uint32_t producer_rank;
    uint32_t flags;

    uint64_t content_hash_lo;
    uint64_t content_hash_hi;

    uint32_t pinned_slot;
    uint64_t pageable_ref;
    uint64_t nvme_ref;

    uint64_t created_ns;
    uint64_t sealed_ns;
    uint64_t last_access_ns;
};
```

## Object States

```text
MISSING
CREATING
VALID_PINNED
VALID_PAGEABLE
VALID_NVME
VALID_MULTI_TIER
SEALED
DELETED
ERROR
```

An object may migrate between physical locations without changing identity.

Example:

```text
object_id 998812
  VALID_PINNED       slot=33
  VALID_PAGEABLE     pageable_ref=120
  VALID_NVME         nvme_ref=0xabc000
  EXPORT_IN_FLIGHT   export_refcount++
```

## Immutability

After a model version is sealed, canonical objects for that version are immutable.

New training updates must create a new model version. They must not mutate objects belonging to an old sealed version.

## Relationship to v1 Slots

v1 physical slot state remains necessary:

```text
FREE
RESERVED_D2H
D2H_IN_FLIGHT
PINNED_VALID
DRAIN_IN_FLIGHT
READBACK_IN_FLIGHT
EXPORT_IN_FLIGHT
```

But slots are now just physical storage backing canonical objects.

## Data Model Rule

Rank-local tensor records should point to canonical objects:

```text
RankTensorRecord {
  rank_id
  local_tensor_id
  local_version
  canonical_object_id
  local_status
}
```

The daemon's authoritative location table should be object-centric, not rank-centric, for model state.
