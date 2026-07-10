# v2 Test Plan

## 1. Job Namespace Tests

### Same job name, different job IDs

Create two jobs with the same `job_name` and different `job_id`.

Expected:

```text
canonical keys do not collide
manifests are separate
quotas are separate
```

### Same job ID, different launch epoch

Simulate job restart.

Expected:

```text
old leases rejected
new launch_epoch gets fresh namespace
old unsealed objects cleaned or retained according to policy
```

## 2. Canonical Dedup Tests

### DP-equivalent attach

DP=2, same param/shard key, different DP rank.

Expected:

```text
rank0 NEED_D2H_CREATE
rank1 ATTACHED_EXISTING
host bytes stored once
D2H bytes reduced by 50%
```

### TP-distinct no attach

Same param, different TP rank.

Expected:

```text
separate canonical objects
no dedup
```

### Expert-distinct no attach

Same layer, different expert_id.

Expected:

```text
separate canonical objects
```

### Hash mismatch

Enable hash-verified mode and intentionally corrupt one replica.

Expected:

```text
hash mismatch detected
canonical attach rejected or error raised
latest valid object not overwritten
```

## 3. Creating Race Tests

### Wait for creator

Two ranks request same missing object at nearly same time.

Expected:

```text
one creator
one waiter
waiter attaches after object becomes valid
```

### Duplicate candidate under pressure

Set policy `duplicate_candidate_on_pressure` and simulate high GPU pressure.

Expected:

```text
both ranks may D2H
first committed object wins
loser candidate discarded safely
```

## 4. Manifest Tests

### Seal success

All expected objects are present.

Expected:

```text
SealModelVersion returns OK
manifest state SEALED
latest sealed pointer updates atomically
```

### Seal fail on missing

One expected tensor object is missing.

Expected:

```text
seal fails when fail_if_missing=true
manifest remains OPEN/ERROR
rollout cannot pull this version
```

### Rollout cannot pull mutable version

Try pulling from POLICY_TRAINABLE or unsealed version.

Expected:

```text
request rejected
```

## 5. Export Tests

### Pull from pinned

Object located in pinned slot.

Expected:

```text
export_refcount increments
transfer succeeds
slot not recycled during export
export_refcount decrements after completion
```

### Pull from pageable

Object located in pageable DRAM.

Expected:

```text
pageable -> export buffer
transfer succeeds
```

### Pull from NVMe

Object located on NVMe.

Expected:

```text
NVMe -> export buffer
transfer succeeds
```

### Export cancellation

Kill rollout engine mid-transfer.

Expected:

```text
export lease eventually releases
object remains valid
no slot leak
```

## 6. Quota Tests

### Per-job pinned quota

Set small max pinned quota for job A.

Expected:

```text
job A requests above quota are rejected/throttled
job B unaffected
```

### Shared arena fairness

Run two jobs competing for same NUMA arena.

Expected:

```text
manager enforces per-job reservations/backpressure
one job cannot starve the other indefinitely
```

## 7. End-to-End RL Sync Test

Setup:

```text
trainer job creates policy_rollout version N
rollout engine pulls version N
trainer creates version N+1
rollout engine diffs and pulls changed objects only
```

Expected:

```text
rollout cache key includes job/version/object/hash
no stale object reused incorrectly
changed tensors update
unchanged tensors not re-pulled
```

## 8. Metrics Validation

Verify counters:

```text
canonical_attach_existing_total
canonical_dedup_bytes_saved
canonical_d2h_bytes_avoided
manifest_sealed_total
rollout_pull_bytes_total
export_inflight_bytes
job_pinned_bytes
job_nvme_bytes
```
