# v2 Canonical Model-State Update — Implementation Guide

This document describes what the v2 update pack asked for and **how it is
implemented** in this repository. v2 is a strictly **additive** layer over the
tested v1 offload runtime: with `manager.enable_canonical_store: false` (or
simply never calling the v2 RPCs) the daemon behaves exactly as v1.

Start with the design narrative in [`update_v2/`](update_v2/):
`00_UPDATE_OVERVIEW.md` → `01_CANONICAL_MODEL_STATE_STORE.md` →
`02_DEDUPLICATION_POLICY.md` → `03_ROLLOUT_PULL_AND_EXPORT.md` →
`04_MULTI_JOB_NAMESPACE_AND_QUOTAS.md`, plus `MIGRATION_FROM_V1.md`,
`LIBFABRIC_EXPORT_BACKEND.md`, and `IMPLEMENTATION_MILESTONES_V2.md`.

## What v2 adds

1. **Canonical model-state object store** — offloaded tensors become immutable,
   versioned canonical objects keyed by
   `(tenant_id, job_id, launch_epoch, model_role, model_version, param + partition coords)`.
2. **Deduplication across replicated ranks** — DP/replica axes are excluded from
   the key, so equivalent ranks attach to one object instead of each storing a
   copy (semantic-trusted fast path; optional hash-verified).
3. **Sealed manifests + rollout pull** — a model version is sealed into an
   immutable manifest; rollout engines pull ONLY sealed versions over a
   transport backend (TCP/file debug, libfabric/north_comm for production).
4. **Multi-job namespace + quotas** — all metadata is job-scoped; per-job
   pinned/pageable/nvme/inflight accounting with optional hard quotas.

## Key implementation decision: reuse v1 tiering

A canonical object does **not** re-implement tiering. Its bytes are backed by
the existing v1 slot/lease/drain/NVMe machinery through a **synthetic
`tensor_id`** (`canonical_tid(object_id) = (1<<63) | object_id`). The create
path mirrors `handle_request_offload`'s slot reservation; the rank then drives
the unchanged `MarkD2HSubmitted → (D2H) → MarkD2HComplete` opcodes against the
synthetic tid, which transitions the slot to `PINNED_VALID`, publishes
`locations_[synthetic_tid]`, and may enqueue a background drain — all exactly as
in v1. `CommitCanonicalObject` only finalizes object metadata (state, content
hash) and resolves duplicate-candidate / hash-verify races. Drain to pageable,
NVMe spill, and readback therefore apply to canonical objects for free; export
staging reads whatever tier the object currently lives in.

## Design doc → code map

| Design doc | Implementation |
|---|---|
| Canonical key / object (`01`) | `abi/offload_canonical_abi.hpp`, `src/common/protocol_v2.h`, `src/daemon/canonical.h` (`CanonicalObject`, `canonical_tid`) |
| RPC schema | `rpc/offload_canonical.proto` → hand-rolled codec in `src/common/wire_v2.{h,cpp}` (no protobuf link; same reason as v1) |
| Attach-or-create dedup (`02`) | `OffloadDaemon::handle_request_canonical_evict` + `handle_commit_canonical_object` (`src/daemon/canonical.cpp`) |
| Sealed manifests, pull/export (`03`) | `handle_seal_model_version` / `handle_get_latest_sealed_version` / `handle_get_manifest` / `handle_pull_tensor` + `stage_object_for_export` (`src/daemon/canonical_rollout.cpp`) |
| Multi-job namespace + quotas (`04`) | `JobRecord` + `handle_register_job` + `quota_allows_pinned` (`src/daemon/canonical.{h,cpp}`); config in `src/daemon/config.{h,cpp}` |
| Libfabric export backend | `src/daemon/export_backend.{h,cpp}` (TCP/file) + `src/daemon/export_libfabric.cpp` (north_comm, `-DOFLD_WITH_LIBFABRIC`) |
| Python API (`canonical_python_api.md`) | `src/python/bindings.cpp` (Context methods) + `python_api/fastoffload/__init__.py` (`Off.canonical_key/canonical_evict/seal/promote`, `DedupPolicy`, `ModelRole`) + `_rollout.py` (`RolloutWeightClient`) + `_wire.py` |
| Multi-job identity | `JobUID{tenant_id, job_id, launch_epoch}`; `job_name` is display-only. Relaunch → fresh `launch_epoch` → distinct namespace |
| State machine (`diagrams/state_machine_v2.mmd`) | object states `OBJ_*` in the ABI; slot states unchanged from v1 |

## Wire protocol

Seven additive opcodes (`src/common/protocol.h`, `OpCode` 20–26):
`RegisterJob`, `RequestCanonicalEvict`, `CommitCanonicalObject`,
`SealModelVersion`, `GetLatestSealedVersion`, `GetManifest`, `PullTensor`.
They ride the same `[magic][opcode][flags][len][body]` frame as v1. The daemon
serves them over the existing UDS **and** an optional TCP control endpoint
(`manager.control_tcp_port` / `offloadd --tcp-port N`) so remote rollout engines
can reach the manifest/pull RPCs. TCP frames never carry SCM_RIGHTS fds, so
`RegisterRank` (which returns arena fds) remains local-only.

## Rollout pull flow

```
rollout ── GetLatestSealedVersion(job, POLICY_ROLLOUT) ──▶ daemon
rollout ── GetManifest(job, role, version) ─────────────▶ daemon
rollout ── (diff manifest vs local cache by object_id+hash)
for each changed object:
  rollout ── PullTensor(object_id, transport, target) ──▶ daemon
        daemon: export_refcount++, OBJ_* → EXPORT_IN_FLIGHT,
                stage bytes from pinned/pageable/nvme,
                transport send (TCP/file/libfabric), export_refcount--
```

The daemon is a **request-triggered sender**: for TCP the rollout client opens a
receiver socket and passes `host:port`; the daemon connects back and streams an
`OFEX`-framed payload (`[magic][ver][object_id][nbytes][hash_lo][hash_hi]` +
bytes). For libfabric the descriptor is `<nic>|<ib_address_hex>` of the client's
north_comm listener.

## NexTrainer integration surface

The RL runtime (`../NexTrainer`) drives v2 at two points in the training loop:

1. **After a training step, when offloading model state** — build a key per
   parameter shard and canonical-evict it:

   ```python
   import fastoffload as fo
   from megatron.core import parallel_state as mpu   # NexTrainer megatron path

   with fo.offload_context(daemon_addr="unix:///tmp/fastoffload.sock",
                           device=f"cuda:{local_rank}",
                           job_name=cfg.job_name,
                           scheduler_job_id=os.environ.get("SLURM_JOB_ID"),
                           dedup_policy=fo.DedupPolicy(mode="semantic_trusted")) as off:
       for name, param in bridge.export_weights(models):   # HF-named FQNs
           key = off.canonical_key(
               model_role="policy_rollout", model_version=global_step,
               param_name=name, tensor=param,
               pp_rank=mpu.get_pipeline_model_parallel_rank(),
               tp_rank=mpu.get_tensor_model_parallel_rank(),
               ep_rank=mpu.get_expert_model_parallel_rank(),
               expert_id=-1)                                # dp_rank excluded
           off.canonical_evict(param, key, destructive=True)
       off.promote_rollout_version(global_step)             # seal + publish
   ```

   Because `dp_rank` is not part of the key, the many DP replicas of each shard
   dedup to one canonical object and only one rank pays the D2H.

2. **On the rollout/inference engine** — pull the sealed version:

   ```python
   client = fo.RolloutWeightClient(
       daemon_addr="tcp://trainer-node:19090",
       job_id=job_id, launch_epoch=launch_epoch, model_role="policy_rollout")
   version = client.get_latest_sealed_version()
   manifest = client.get_manifest(version)
   for entry in client.diff_local(manifest):
       raw = client.pull_tensor(entry, version, transport="tcp")  # or "libfabric"
       load_into_engine(entry["key"]["param_fqn_debug"], raw)
   ```

   The reference pull pattern is `../NorthCheckpoint` (WeightClient/
   WeightProvider); `RolloutWeightClient` layers version discovery, manifest
   diff, and content-hash caching on top of the same north_comm transport.

## Metrics

All v2 counters are exposed through `GetStats` (`src/common/metrics.{h,cpp}`):
`canonical_objects_created_total`, `canonical_attach_existing_total`,
`canonical_dedup_bytes_saved`, `canonical_d2h_bytes_avoided`,
`canonical_hash_mismatch_total`, `manifest_sealed_total`,
`rollout_pull_bytes_total`, `export_inflight_bytes`, `export_from_{pinned,
pageable,nvme}_total`, `quota_rejections`, and more.

## Tests

- `tests/cpp/test_wire_v2.cpp` — v2 codec round-trips.
- `tests/cpp/test_canonical.cpp` — live-daemon integration: job namespace/epoch,
  DP-attach vs TP/expert-distinct, creating race (wait + duplicate-candidate),
  hash-verified mismatch reject, seal (+ fail-on-missing) + promote + manifest,
  file & TCP transport pull, per-job quota.
- `tests/cpp/test_config.cpp::test_v2_config` — overlay config parsing.
- `tests/python/test_canonical_e2e.py` — GPU end-to-end: create + DP-attach,
  seal+promote, `RolloutWeightClient` pull byte-exact + cache hit.
- Plan: `tests/TEST_PLAN_V2.md`.

## Build

```
mkdir build && cd build && cmake .. && make -j && ctest      # v1 + v2 C++
../run_tests.sh                                              # + Python e2e
```

Libfabric backend (optional, needs the north_comm build):
```
cmake .. -DOFLD_WITH_LIBFABRIC=ON -DNORTH_COMM_DIR=/path/to/north_comm
```
