# StateSpectre

**A centralized GPU offload and model-state runtime for large-scale LLM training and RL post-training.**

StateSpectre evacuates tensors from GPU VRAM into host memory and NVMe in
milliseconds, then serves them back on demand — and, for RL post-training, turns
those offloaded tensors into a versioned, deduplicated **canonical model-state
store** that rollout/inference engines pull sealed weights from over RDMA.

One node-level daemon (`offloadd`) owns the memory and the metadata; every
training rank attaches to it through a thin agent and a Python package
(`state_spectre`). The daemon is CUDA-free; ranks own all CUDA.

---

## Why

Three recurring costs in large training / RL jobs, one runtime:

| Problem | What StateSpectre does |
|---|---|
| **Freeing VRAM is slow and blocking** — offloading weights/optimizer/KV-cache stalls the training step | D2H into pre-registered pinned arenas; VRAM is released the instant the copy's CUDA event fires (~1.66 s for 80 GB on PCIe Gen5). Spill to pageable DRAM / NVMe is asynchronous and never gates the release. |
| **Redundant copies across replicas** — every data-parallel rank offloads an identical weight shard | Canonical objects keyed by everything *except* the replicated axis; equivalent ranks attach to one object, so only one rank pays the D2H. |
| **Weight sync to rollout engines is a full-model, blocking round-trip** | Sealed, immutable model-version manifests; rollout engines pull only sealed versions, diff against a content-hash cache, and pull just what changed — over TCP or libfabric/RDMA. |

## Capabilities

- **Fast destructive eviction & restore** — `off.evict(tensor)` / `handle.restore()` with real storage replacement, byte-identical round-trips.
- **Tiered residency** — pinned host RAM → pageable DRAM → NVMe (O_DIRECT, striped across drives, `io_uring` or `pwrite`), with background drain and prefetch.
- **NUMA-aware** — one fd-backed pinned arena per NUMA node, GPU→node affinity, per-GPU allocation windows.
- **Multi-process safe** — daemon-owned leases, epochs, and crash recovery; a lock-free shared-memory completion ring on the hot path (no syscall per completion).
- **Canonical model-state store** — job-scoped (`tenant / job / launch_epoch`) versioned objects, semantic or hash-verified dedup across replicated ranks.
- **Sealed manifests + rollout pull** — immutable model versions; a `RolloutWeightClient` discovers, diffs, and pulls sealed weights. Transports: TCP, file, and libfabric (validated on real cross-node RDMA).
- **Observable** — a fixed metric set over `offload-stat` (human / Prometheus / watch) and a structured JSON trace.

## Architecture at a glance

```
  training rank 0..N                          offloadd  (one per node, CUDA-free)
 ┌───────────────────────┐                   ┌────────────────────────────────────┐
 │ state_spectre (Python)  │   AF_UNIX (RPC,    │ per-NUMA pinned arenas (memfd+mbind)│
 │   off.evict / restore │   SCM_RIGHTS fds)  │ slot allocator · leases · locations │
 │   canonical_evict     │◀──────────────────▶│ canonical object + manifest store   │
 │ OffloadAgent (C++)    │   + shared control │ drain/readback workers · NVMe engine│
 │   cudaHostRegister    │   region (slot     │ heartbeat monitor · crash recovery  │
 │   D2H/H2D + events     │   table + rings)  │ export backends (TCP/file/libfabric)│
 └───────────────────────┘                   └────────────────────────────────────┘
                                                        ▲  TCP control port
                                                        │  (remote rollout engines)
                                    rollout / inference engine — RolloutWeightClient
```

## Quickstart

```bash
# Build the daemon, agent, tools, and tests.
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

# Build the Python extension (matches your libtorch ABI).
cd ../python_api && python setup.py build_ext --inplace

# Run the daemon (smoke config: one arena on NUMA 0 for GPU 0).
../build/offloadd --smoke-arena-mb 8192 --numa 0 --gpu 0 --socket /tmp/state_spectre.sock
```

```python
import torch, state_spectre as ss

with ss.offload_context(daemon_addr="unix:///tmp/state_spectre.sock", device="cuda:0") as off:
    x = torch.randn(8192, 8192, device="cuda")
    h = off.evict(x)          # frees VRAM once the D2H event completes
    x = h.restore()           # byte-identical
```

RL post-training weight sync (trainer seals a version; rollout pulls it):

```python
# Trainer
with ss.offload_context(..., job_name="qwen32b_grpo") as off:
    key = off.canonical_key(model_role="policy_rollout", model_version=step,
                            param_name=name, tensor=w)   # dp axis excluded → replicas dedup
    off.canonical_evict(w, key)
    off.promote_rollout_version(step)                    # seal + publish

# Rollout engine (remote)
client = ss.RolloutWeightClient(daemon_addr="tcp://trainer:19090",
                                job_id=job_id, launch_epoch=epoch)
for entry in client.diff_local(client.get_manifest(client.get_latest_sealed_version())):
    client.pull_tensor(entry, version, transport="libfabric")
```

## Documentation

| Doc | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The whole system: daemon, agent, canonical store, transports, and the design decisions behind them |
| [docs/USAGE.md](docs/USAGE.md) | Build, run, configure, the Python & C++ APIs, observability, and tests |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Measured throughput on the target hardware (D2H/H2D, NVMe sweep, NUMA) |
| [docs/design/](docs/design/) | Deeper design references: memory tiering, race conditions, dedup, rollout pull, libfabric backend, ABI & RPC notes |
| [CLAUDE.md](CLAUDE.md) | Repository layout, build/test workflow, and conventions for developing the codebase |

## Status

Implemented and tested on real hardware (8× H20, NVMe, multi-node RoCE/RDMA).
The C++ test suite (`ctest`), Python end-to-end / multi-process / LLM-style
workload tests, and a real cross-node libfabric weight-pull all pass; see
[docs/USAGE.md](docs/USAGE.md#tests). Requires Linux (`memfd_create`, `mbind`),
CUDA, and — for the RDMA export backend — libfabric via `north_comm`.
