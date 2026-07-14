# Python API Additions for Canonical Model-State Offload

## Design Goal

Keep the v1 Python interface usable for arbitrary tensor offload, while adding canonical model-state semantics for RL post-training.

## Job Context

The offload context must now be job-aware.

```python
import state_spectre as ss

with ss.offload_context(
    daemon_addr="unix:///tmp/ss.sock",
    device="cuda:0",
    job_name="qwen32b_grpo_math",
    tenant_id=0,
    scheduler_job_id=os.environ.get("SLURM_JOB_ID"),
) as off:
    ...
```

The daemon returns:

```python
off.job_id
off.launch_epoch
off.job_name
```

`job_name` is human-readable. The actual identity is `(tenant_id, job_id, launch_epoch)`.

## Canonical Key Builder

```python
key = off.canonical_key(
    model_role="policy_trainable",
    model_version=global_step,
    param_name="layers.12.mlp.w1.weight",
    param_id=3012,
    tensor=tensor,
    pp_rank=pp_rank,
    tp_rank=tp_rank,
    ep_rank=ep_rank,
    etp_rank=etp_rank,
    expert_id=-1,
    shard_offset=0,
    shard_nbytes=tensor.nbytes,
)
```

Default excluded axis:

```text
dp_rank
```

because DP is replicated.

## Canonical Evict

```python
h = off.evict(
    tensor,
    canonical_key=key,
    dedup="attach_or_create",
    destructive=True,
    stream=torch.cuda.current_stream(),
)
```

Possible outcomes:

```text
ATTACHED_EXISTING:
    no D2H performed; local tensor may be invalidated after safe point

NEED_D2H_CREATE:
    rank performs GPU -> pinned D2H and creates canonical object

WAIT_FOR_CREATOR:
    another rank is creating the object; wait or fallback depending on policy

DUPLICATE_CANDIDATE:
    rank performs local D2H into temporary candidate to release VRAM quickly
```

## Dedup Policy

```python
ss.DedupPolicy(
    mode="semantic_trusted",       # disabled | semantic_trusted | hash_verified | sampled_hash
    creating_policy="wait",        # wait | duplicate_candidate_on_pressure
    cross_job_dedup=False,
)
```

Attach to context:

```python
with ss.offload_context(..., dedup_policy=policy) as off:
    ...
```

## Sealing a Model Version

After trainer has committed all canonical objects for a rollout-visible version:

```python
off.seal_model_version(
    model_role="policy_rollout",
    model_version=step,
    fail_if_missing=True,
)
```

This creates an immutable manifest.

## Promoting Rollout Version

Optional convenience API:

```python
off.promote_rollout_version(model_version=step)
```

Equivalent to:

```text
seal POLICY_ROLLOUT version step
atomically set latest sealed rollout pointer to step
```

## Rollout Client API

A rollout engine can use:

```python
client = ss.RolloutWeightClient(
    daemon_addr="tcp://trainer-node:19090",
    job_id=job_id,
    launch_epoch=launch_epoch,
    model_role="policy_rollout",
)

version = client.get_latest_sealed_version()
manifest = client.get_manifest(version)
changed = client.diff_local(manifest)

for entry in changed:
    client.pull_tensor(
        entry,
        transport="libfabric",
        destination=rollout_weight_buffer(entry),
    )
```

## Local Cache

Rollout engines should cache by:

```text
(job_id, launch_epoch, model_role, model_version, object_id, content_hash)
```

not just by parameter name.

## Error Behavior

`off.evict(..., canonical_key=...)` should reject if:

```text
job is not registered
model_version is already sealed
canonical key belongs to another job
shape/dtype mismatch existing object
quota exceeded
hash mismatch in verified mode
```

## Backward Compatibility

Existing v1 call remains valid:

```python
h = off.evict(tensor, destructive=True)
```

This creates a rank-local offload handle without canonical dedup or rollout manifest inclusion.

For model states, users should pass `canonical_key`.
