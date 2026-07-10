# Multi-Job Namespace and Quotas

## Problem

Multiple RL jobs may share one node-level offload manager. Therefore, all metadata, object keys, manifests, quotas, metrics, and cleanup logic must be job-scoped.

## Job Identity

Use both a stable ID and human-readable name.

```cpp
struct JobKey {
    uint64_t tenant_id;
    uint64_t job_id;
    uint64_t launch_epoch;
    char     job_name[128];
};
```

Rules:

```text
job_id + launch_epoch is identity-critical
job_name is display/debug metadata only
```

Do not use job name alone for uniqueness.

## Why launch_epoch?

Schedulers may relaunch a job with the same job ID/name after failure. `launch_epoch` distinguishes old stale leases/objects from the current process group.

## Namespace Rule

All these must include job scope:

```text
leases
rank sessions
canonical objects
model manifests
pull/export requests
quotas
metrics
NVMe allocation records
cleanup ownership
```

## Canonical Dedup Scope

Default dedup scope:

```text
same tenant_id
same job_id
same launch_epoch
same model_role
same model_version
same canonical tensor key
```

Cross-job dedup is disabled by default.

## Resource Accounting

Track per job:

```text
pinned_bytes
pageable_bytes
nvme_bytes
inflight_d2h_bytes
inflight_h2d_bytes
inflight_export_bytes
canonical_object_count
slot_count
reservation_failures
quota_rejections
```

## Quotas

Example policy:

```yaml
jobs:
  default:
    max_pinned_bytes: 256GiB
    max_pageable_bytes: 1TiB
    max_nvme_bytes: 8TiB
    max_inflight_d2h_bytes: 160GiB
    priority: normal
```

Quota enforcement points:

```text
RequestOffloadLease
RequestCanonicalEvict
CreateCanonicalObject
DrainToPageable
DrainToNvme
PrepareExport
```

## Cleanup

When a job terminates:

```text
1. mark job state TERMINATING
2. reject new leases/pulls unless allow_graceful_export=true
3. wait for or abort inflight transfers
4. decrement/release objects scoped to job
5. free pinned/pageable/NVMe allocations
6. remove sealed manifests unless retention policy keeps them
```

Retention policy may keep sealed rollout versions for debugging or delayed rollout catch-up.

## Job State

```text
REGISTERED
RUNNING
DRAINING
TERMINATING
TERMINATED
ERROR
```

## Required Metrics

All metrics should be labeled by:

```text
tenant_id
job_id
job_name
launch_epoch
model_role, where applicable
```
