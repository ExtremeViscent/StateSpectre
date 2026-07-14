# Architecture

StateSpectre is one node-level daemon plus a per-rank agent. This document
describes the whole system as it runs; deeper design references live in
[docs/design/](design/).

```
                       ┌──────────────────────────────────────────────────┐
   rank process 0      │              offloadd  (CUDA-free)                 │
 ┌────────────────────┐│  ┌──────────────────────────────────────────────┐ │
 │ fastoffload (Python)││  │ per-NUMA pinned arenas (memfd + mbind)         │ │
 │  off.evict/restore  ││  │ slot allocator (per-GPU windows + overflow)    │ │
 │  canonical_evict    ││  │ lease / location / session tables              │ │
 │  ↕ pybind11         ││  │ canonical object + manifest store (v2)         │ │
 │ OffloadAgent (C++)  ││  │ drain + readback worker pools                  │ │
 │  cudaHostRegister   ││  │ NVMe engine (io_uring / pwrite, O_DIRECT)      │ │
 │  D2H/H2D + events   ││  │ heartbeat monitor + lease/crash recovery       │ │
 │  progress thread    ││  │ completion-ring poller                         │ │
 └─────────┬───────────┘│  │ export backends (TCP / file / libfabric)       │ │
           │            │  │ metrics + JSON trace                           │ │
           │            │  └──────────────────────────────────────────────┘ │
           │  AF_UNIX control plane (SCM_RIGHTS fd passing)                   │
           │  + shared control region (slot table + per-rank completion rings)│
           └──────────────────────────────────────────────────────────────────┘
        ranks 1..N attach the same fds + control region;  optional TCP control
        port serves the canonical/rollout RPCs to remote inference engines.
```

## Division of responsibility

The rank owns **all CUDA** — streams, events, `cudaHostRegister`, and the D2H/H2D
copies — because it already owns the PyTorch allocator and stream state. The
daemon owns **memory, metadata, and lifetime** and never links CUDA. This avoids
CUDA-IPC lifetime hazards, cross-process GPU-pointer ownership, and daemon-side
CUDA contexts. The two communicate over an AF_UNIX control plane (a hand-rolled
binary RPC with `SCM_RIGHTS` fd passing) plus a shared, fd-backed control region.

## Data plane: arenas, slots, tiers

- **Arenas.** One `memfd`-backed pinned arena per NUMA node, NUMA-localized with
  `mbind` + first-touch. Ranks `mmap` the same fds and register each arena once
  with `cudaHostRegister` (whole-arena, not per-copy). GPU→NUMA affinity comes
  from sysfs; each arena is carved into per-GPU allocation *windows* plus an
  overflow window.
- **Slots & leases.** The daemon allocates contiguous slot runs from a window,
  hands the rank a lease + arena offset, and the rank does the copy. Slot state
  lives in the shared control region as atomics the daemon writes and ranks read.
- **Eviction is the critical path; drain is not.** On `evict`, the rank
  `cudaMemcpyAsync` D2H into the slot and records a CUDA event. **VRAM is
  released the moment that event completes** — measured ~1.66 s for 80 GB on
  PCIe Gen5. Migration to the cold tiers (pageable DRAM, then NVMe) is
  asynchronous background work that never gates the release. See
  [design/memory-tiering.md](design/memory-tiering.md).
- **NVMe.** O_DIRECT, striped across drives (one dir per device), with an
  `io_uring` or `pwrite` engine; block size and queue depth are tunable
  (16 MB / QD 16 is the measured sweet spot — see [PERFORMANCE.md](PERFORMANCE.md)).
- **Restore/prefetch.** `restore` resolves the tensor's current tier from the
  location table, reads it back into a pinned slot if cold, then H2D into a
  caller buffer. `prefetch` warms it ahead of use.

## Control plane

A fixed-layout little-endian codec (`src/common/wire*.{h,cpp}`) mirrors the
message structs in `protocol.h` / `protocol_v2.h`; `rpc/*.proto` is the schema
of record but is **not** linked (libtorch bundles its own protobuf; a second one
in the agent process risks ODR clashes). One connection thread per client
decodes → handler (under one mutex) → encodes the reply. Long IO in workers runs
without the lock (copy metadata out, unlock, do IO, relock to commit).

**Completion reporting** has two paths: a lock-free single-producer/consumer
**shared-memory ring** per rank (no syscall per completion, polled by the
daemon) and a batched RPC fallback. **Crash recovery**: ranks heartbeat; on
timeout the daemon invalidates the session, reclaims in-flight leases, and keeps
safely-persisted data. Rank epochs fence stale sessions after a relaunch. See
[design/race-conditions.md](design/race-conditions.md) and
[design/rpc-semantics.md](design/rpc-semantics.md).

## Canonical model-state store (RL post-training)

For RL post-training, offloaded tensors are promoted to immutable, versioned
**canonical objects**. This layer is strictly additive — disabled, the daemon is
exactly the tiering runtime above.

- **Identity.** A canonical key is
  `(tenant_id, job_id, launch_epoch, model_role, model_version, param + partition
  coords + shard range + dtype/shape)`. It includes every axis that affects the
  bytes (pp/tp/ep/etp/expert, shard offset) and **excludes** the replicated axis
  (data-parallel rank). `job_name` is display-only; `launch_epoch` fences a
  relaunched job into a fresh namespace. See [design/canonical-store.md](design/canonical-store.md).
- **Reuse of the tiering engine.** A canonical object is backed by the v1
  slot/lease/drain/NVMe machinery through a synthetic `tensor_id`
  (`1<<63 | object_id`); the rank drives the unchanged D2H path to fill it, and
  drain/NVMe/readback apply for free. `CommitCanonicalObject` only finalizes
  object metadata and resolves races.
- **Deduplication.** Because the key excludes the DP axis, equivalent ranks
  produce the same key. The first rank creates the object (`NEED_D2H_CREATE`);
  the rest `ATTACH_EXISTING` with **no D2H**. Modes: semantic-trusted (fast
  path), hash-verified (D2H into a candidate, compare content hash, reject a
  corrupt replica), or disabled. A creating-race is resolved by
  wait-for-creator or, under GPU pressure, an allow-duplicate-candidate policy
  where the first commit wins. See [design/deduplication.md](design/deduplication.md).
- **Sealed manifests.** `SealModelVersion` freezes all objects of a
  `(job, role, version)` into an immutable manifest and (optionally) atomically
  promotes it to the latest-sealed pointer. Rollout engines see a version only
  after it is sealed.
- **Multi-job & quotas.** Every table, object, manifest, and metric is
  job-scoped. Per-job accounting (pinned / pageable / nvme / inflight) with
  optional hard quotas gates reservations. See [design/multi-job-and-quotas.md](design/multi-job-and-quotas.md).

## Rollout pull & export

Rollout / inference engines pull sealed weights, over the daemon's optional TCP
control port (so they can be remote):

```
rollout ── GetLatestSealedVersion(job, role) ─▶ daemon
rollout ── GetManifest(job, role, version) ───▶ daemon
        ── diff manifest vs local cache (object_id + content_hash) ──
for each changed object:
  rollout ── PullTensor(object_id, transport, target) ─▶ daemon
          daemon: export_refcount++, stage bytes from pinned/pageable/nvme,
                  send via transport, export_refcount-- (lease protects lifetime)
```

The daemon is a **request-triggered sender**. Export leases (`export_refcount`)
keep an object's backing alive across drain/recycle while a transfer is in
flight. Transports: **TCP** and **file** (debug/portable), and **libfabric**
(via `north_comm`, validated on real cross-node RDMA — see
[design/libfabric-backend.md](design/libfabric-backend.md)). The Python
`RolloutWeightClient` implements discover → manifest → diff → pull with a
content-hash cache so unchanged tensors are not re-pulled. See
[design/rollout-pull-and-export.md](design/rollout-pull-and-export.md).

## Interfaces

- **Python** (`fastoffload`): `offload_context()`, `off.evict/copy/restore`,
  `evict_many/restore_many`, and the canonical layer `canonical_key`,
  `canonical_evict`, `DedupPolicy`, `seal_model_version` /
  `promote_rollout_version`, `RolloutWeightClient`. See
  [python_api/PYTHON_API.md](../python_api/PYTHON_API.md) and
  [python_api/canonical_python_api.md](../python_api/canonical_python_api.md).
- **C++** (`src/agent/agent.h`): `OffloadAgent` for embedding without Python —
  raw device pointers and stream handles, no torch dependency.
- **Observability**: `offload-stat` (human / Prometheus / watch) over `GetStats`;
  structured JSON trace via `OFLD_TRACE`.

## Wire protocol summary

Frame: `[u32 magic][u16 opcode][u16 flags][u32 len][payload]`. v1 opcodes cover
register/heartbeat/offload/D2H/prefetch/H2D/release/query/batch/stats/shutdown;
v2 adds `RegisterJob`, `RequestCanonicalEvict`, `CommitCanonicalObject`,
`SealModelVersion`, `GetLatestSealedVersion`, `GetManifest`, `PullTensor`
(opcodes 20–26), riding the same frame. The daemon serves them over the UDS and,
when a TCP control port is configured, over TCP as well (TCP frames never carry
fds, so arena-fd passing stays local-only).
