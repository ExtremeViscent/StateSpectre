# v2 Update Overview

## Problem Context

The v1 offload manager solves fast evacuation of tensors from GPU VRAM into pinned host memory, followed by optional background movement to pageable DRAM and NVMe.

For LLM RL post-training, two new problems appear:

1. **Redundant model states across ranks**: for replicated axes such as data parallelism, many ranks hold equivalent bytes for the same logical parameter shard.
2. **Remote rollout sync**: rollout engines need to pull weights from the trainer side, often via a direct P2P/libfabric tensor transport.
3. **Multiple jobs sharing one manager**: a node-level offload manager may serve several RL jobs simultaneously, so all metadata must be job-scoped.

## Key v2 Abstraction

Promote offloaded tensors into immutable, versioned canonical objects:

```text
CanonicalObject = bytes for one logical model-state shard at one sealed model version
```

The offload system changes from:

```text
rank-local tensor store
```

to:

```text
versioned canonical model-state object store
```

## v2 Identity Stack

Use this namespace stack:

```text
tenant_id
  -> job_id + launch_epoch
    -> model_role
      -> model_version
        -> canonical tensor key
```

Where `job_name` is human-readable metadata only. It must not be used as the sole uniqueness key.

## Model Role

A single RL job may have several model roles:

```text
POLICY_TRAINABLE
POLICY_ROLLOUT
POLICY_REFERENCE
REWARD_MODEL
VALUE_MODEL
AUXILIARY
```

Each role has independent model versions and manifests.

## Rollout Sync Rule

Rollout engines only pull from:

```text
(job_id, model_role=POLICY_ROLLOUT, model_version=SEALED)
```

not from mutable trainable weights.

## Implementation Order

Recommended additive rollout:

1. Add job namespace and quotas to v1 metadata.
2. Add canonical keys and object table.
3. Add local semantic dedup with attach-or-create.
4. Add sealed manifests.
5. Add debug TCP/file pull transport.
6. Add libfabric export backend.
7. Optimize with RDMA-read/zero-copy export leases.
