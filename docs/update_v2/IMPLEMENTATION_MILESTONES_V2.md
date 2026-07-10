# Implementation Milestones v2

## Milestone 1: Job Namespace

Deliverables:

```text
RegisterJob RPC
job_id + launch_epoch attached to rank sessions
job-scoped logs/metrics
soft quota accounting
```

Acceptance:

```text
two jobs with same job_name do not collide
stale launch_epoch leases are rejected
```

## Milestone 2: Canonical Object Table

Deliverables:

```text
CanonicalTensorKey
CanonicalObjectEntry
object_id allocator
object -> physical location mapping
```

Acceptance:

```text
objects can be created, committed, looked up, refcounted, deleted
```

## Milestone 3: Attach-or-Create Dedup

Deliverables:

```text
RequestCanonicalEvict
CommitCanonicalObject
semantic trusted dedup within job scope
creating-object wait policy
```

Acceptance:

```text
DP-equivalent tensors attach to one object
TP/EP-distinct tensors do not attach
```

## Milestone 4: Pressure-Aware Duplicate Candidate

Deliverables:

```text
allow_duplicate_candidate
winner/loser candidate resolution
optional sampled hash verification
```

Acceptance:

```text
under simulated high GPU pressure, ranks can release VRAM without waiting for first creator
loser candidate cleanup has no leaks
```

## Milestone 5: Sealed Manifests

Deliverables:

```text
SealModelVersion
GetLatestSealedVersion
GetManifest
manifest state machine
latest sealed pointer
```

Acceptance:

```text
rollout-visible version only updates after successful seal
unsealed versions cannot be pulled
```

## Milestone 6: Debug Export Transport

Deliverables:

```text
PullTensor over TCP/file
export leases/refcounts
dedicated export buffer pool
```

Acceptance:

```text
rollout client pulls manifest and tensors correctly without libfabric
```

## Milestone 7: Libfabric Export Backend

Deliverables:

```text
transport adapter wrapping existing P2P tensor sender
request-triggered send
export buffer registration
transport-specific target descriptors
```

Acceptance:

```text
rollout engine pulls sealed manifest weights over libfabric
object lifetime protected during export
```

## Milestone 8: Optimization

Deliverables:

```text
RDMA read support
zero-copy export from pinned object slots
cross-node object routing
full/sampled hash modes
cross-job content dedup behind explicit policy
```

Acceptance:

```text
same correctness as debug transport
improved export bandwidth and lower staging overhead
```
