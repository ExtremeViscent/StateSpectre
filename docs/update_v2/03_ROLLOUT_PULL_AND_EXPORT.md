# Rollout Pull and Export Integration

## Goal

Rollout engines deployed remotely should proactively pull model weights from trainer-side offload managers. The offload manager should expose sealed canonical objects through a versioned manifest and transport backend such as libfabric.

## Principle

Rollout engines pull from sealed model versions, not mutable trainer state.

```text
rollout -> GetLatestSealedVersion(job_id, role=POLICY_ROLLOUT)
rollout -> GetManifest(job_id, role, version)
rollout -> PullTensor(object_id)
```

## Sealed Manifest

A manifest describes an immutable model snapshot.

```json
{
  "tenant_id": 0,
  "job_id": 912884112,
  "job_name": "qwen32b_grpo_math",
  "launch_epoch": 3,
  "model_role": "POLICY_ROLLOUT",
  "model_version": 1042,
  "state": "SEALED",
  "tensors": [
    {
      "name": "layers.12.mlp.w1.weight",
      "param_id": 3012,
      "object_id": 998812,
      "dtype": "bf16",
      "shape": [14336, 4096],
      "nbytes": 117440512,
      "pp_rank": 1,
      "tp_rank": 3,
      "ep_rank": 0,
      "etp_rank": 0,
      "expert_id": -1,
      "shard_offset": 0,
      "shard_nbytes": 117440512,
      "content_hash": "optional",
      "location_hint": "node3:nvme"
    }
  ]
}
```

## Version Promotion Flow

```text
1. Trainer writes model_role=POLICY_TRAINABLE version N.
2. Required canonical objects are committed.
3. Control plane builds manifest N.
4. Manifest state becomes SEALED.
5. Rollout latest pointer is atomically promoted to N.
6. Rollout engines discover and pull N.
```

Do not promote a version until all required canonical objects are valid.

## Pull Flow

```text
rollout engine:
  latest = GetLatestSealedVersion(job_id, POLICY_ROLLOUT)
  manifest = GetManifest(job_id, POLICY_ROLLOUT, latest)
  diff manifest against local cache
  for changed objects:
      PullTensor(object_id, destination)
```

## Export Path

Trainer-side object location may be:

```text
pinned
pageable DRAM
NVMe
remote node daemon
```

Export server behavior:

```text
if object is pinned and export-safe:
    export directly or stage into export buffer

if object is pageable:
    stage pageable -> export buffer

if object is NVMe:
    read NVMe -> export buffer

then:
    send via libfabric / TCP / file transport
```

## Export Leases

Do not let remote transfer race with slot recycling or drain.

Each export increments:

```text
object.export_refcount++
```

The backing storage or export buffer cannot be released while `export_refcount > 0`.

Physical state extension:

```text
PINNED_VALID -> EXPORT_IN_FLIGHT -> PINNED_VALID
```

or:

```text
COLD_VALID_NVME -> READBACK_TO_EXPORT -> EXPORT_IN_FLIGHT -> COLD_VALID_NVME
```

## Dedicated Export Buffers

Preferred v1 remote-sync path:

```text
canonical object -> dedicated export staging buffer -> libfabric send
```

Avoid exposing arbitrary offload pinned slots directly to rollout engines until export leases and refcounts are proven correct.

## Direct RDMA Read Optimization

Future optimization:

```text
object in registered pinned buffer
rollout obtains rkey/remote address
rollout RDMA-reads directly
```

Only enable when:

```text
object is immutable or export-refcount protected
memory region is registered for libfabric
remote access policy is enforced
lifetime cannot end during RDMA read
```

## Required Metrics

```text
manifest_sealed_total
manifest_seal_failed_total
rollout_get_manifest_total
rollout_pull_tensor_total
rollout_pull_bytes_total
export_inflight_bytes
export_staging_bytes
export_from_pinned_total
export_from_pageable_total
export_from_nvme_total
export_transport_errors_total
rollout_stale_manifest_reject_total
```
