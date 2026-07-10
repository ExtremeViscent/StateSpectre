# Migration from v1 to v2

## Objective

Add canonical object store, multi-job scoping, sealed manifests, and rollout pull/export without breaking existing v1 offload behavior.

## Step 1: Add Job Registration

Add a job registration RPC and make every rank session carry:

```text
tenant_id
job_id
launch_epoch
job_name
```

Update all logs and metrics to include job labels.

Existing v1 offload requests can use an implicit default job if necessary during transition.

## Step 2: Add Job-Scoped Quotas

Add per-job accounting for:

```text
pinned_bytes
pageable_bytes
nvme_bytes
inflight_d2h_bytes
inflight_export_bytes
```

Do not enable hard enforcement until accounting is validated.

## Step 3: Add Canonical Key Type

Add canonical key builder in Python and canonical key structures in daemon/RPC.

Initially, store canonical metadata but do not dedup.

## Step 4: Add Object Table

Add:

```text
CanonicalObjectTable
CanonicalObjectState
object_id allocator
object -> physical location mapping
```

Keep v1 slot table unchanged.

## Step 5: Add Attach-or-Create

Implement:

```text
RequestCanonicalEvict
CommitCanonicalObject
```

Start with:

```text
dedup=disabled
```

Then enable semantic dedup for DP-equivalent known-safe tensors.

## Step 6: Add Sealed Manifests

Implement:

```text
SealModelVersion
GetLatestSealedVersion
GetManifest
```

Rollout engines should only consume SEALED manifests.

## Step 7: Add Debug Pull Transport

Before libfabric, add a simple debug transport:

```text
TCP stream
or
local file path export
```

Validate manifest diff, object lookup, and full tensor correctness.

## Step 8: Add Libfabric Export Backend

Integrate existing direct P2P tensor sender as an export backend behind:

```text
PullTensor(object_id, target_descriptor)
```

Use dedicated export buffers first.

## Step 9: Optimize

After correctness:

```text
allow direct export from pinned canonical object slots
add RDMA read support
add sampled/full hash verification modes
add cross-node object routing
```

## Backward Compatibility

v1 API:

```python
h = off.evict(tensor)
```

continues to create a rank-local object.

v2 API:

```python
h = off.evict(tensor, canonical_key=key, dedup="attach_or_create")
```

creates or attaches to canonical object.
