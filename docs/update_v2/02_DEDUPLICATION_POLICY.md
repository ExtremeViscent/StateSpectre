# Deduplication Policy

## Goal

Save host RAM, pinned memory, NVMe capacity, and D2H bandwidth by canonicalizing redundant model states across equivalent ranks.

## Correctness Rule

Deduplication is legal only when two tensors have the same:

```text
tenant_id
job_id
launch_epoch
model_role
model_version
param identity
partition coordinates
shard range
dtype/layout/shape
```

and differ only in replicated axes such as DP rank.

## Dedup Modes

### 1. Semantic Trusted Dedup

The training framework guarantees equivalence.

```text
same canonical key => same bytes
```

This is the production fast path.

Behavior:

```text
First rank creates canonical object.
Later equivalent ranks attach to the existing object and can release local GPU memory without D2H.
```

### 2. Hash-Verified Dedup

The daemon verifies that equivalent keys have equivalent bytes.

Options:

```text
full hash      safest, expensive
sampled hash   lower overhead, debug/prod compromise
metadata only  fastest, assumes framework correctness
```

Hashing must not be on the VRAM-release critical path. Hash after D2H completion or sample opportunistically.

### 3. Disabled Dedup

All ranks offload independently. Useful for debugging or unsupported sharding layouts.

## Attach-or-Create Flow

Python:

```python
h = off.evict(
    tensor,
    canonical_key=key,
    dedup="attach_or_create",
    destructive=True,
)
```

Daemon behavior:

```text
if canonical object exists and is VALID/SEALED:
    increment refcount
    return ATTACHED_EXISTING

elif canonical object is CREATING:
    either WAIT_FOR_CREATOR or ALLOW_DUPLICATE_CANDIDATE depending on policy

else:
    allocate pinned slot
    return NEED_D2H_CREATE
```

Rank behavior:

```text
ATTACHED_EXISTING:
    no D2H required
    invalidate local tensor once safe

NEED_D2H_CREATE:
    perform normal GPU -> pinned D2H
    report completion
    object becomes VALID
```

## Handling CREATING Object

When another rank is already creating the same canonical object, choose policy:

### WAIT_FOR_CREATOR

```text
rank B waits until rank A finishes D2H
then rank B attaches and invalidates local GPU tensor
```

Saves bandwidth but may delay VRAM release.

### ALLOW_DUPLICATE_CANDIDATE

```text
rank B also D2H copies into a temporary candidate object
first valid object wins
loser candidate discarded or verified
```

Improves local VRAM release latency but consumes transient pinned bandwidth and memory.

Recommended default:

```text
WAIT_FOR_CREATOR for normal pressure
ALLOW_DUPLICATE_CANDIDATE if local GPU memory pressure is critical
```

## Cross-Job Dedup

Disabled by default.

Even if two jobs use the same base model, do not dedup across jobs unless explicitly enabled with content-hash verification and trust policy.

Reason:

```text
job names can collide
versions mean different things across jobs
training mutations may diverge
security/isolation concerns
```

## Required Metrics

```text
canonical_objects_created_total
canonical_attach_existing_total
canonical_attach_wait_total
canonical_duplicate_candidate_total
canonical_dedup_bytes_saved
canonical_d2h_bytes_avoided
canonical_hash_mismatch_total
canonical_stale_version_reject_total
```
