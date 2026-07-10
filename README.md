# Centralized GPU Offload Runtime Handoff Pack

> **This pack is now a complete, tested implementation**, not just a design.
> The C++/CUDA daemon + rank agent + Python API live under `src/` and
> `python_api/`. Start here:
>
> - **[docs/](docs/README.md)** — consolidated documentation
>   - [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — architecture & what was built
>   - [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — measured benchmarks (80 GB evict ≈ 1.66 s on PCIe Gen5)
>   - [docs/USAGE.md](docs/USAGE.md) — build, run, configure, test
>
> Quick start: `mkdir build && cd build && cmake .. && make -j && ctest` — then
> `../run_tests.sh` for the full GPU + Python suite. All seven milestones in
> `impl/IMPLEMENTATION_PLAN.md` are implemented and verified.

---

This bundle contains an implementation-facing design for a centralized GPU tensor offloading system.

The design assumes:

1. The user/runtime can guarantee that destructively offloaded tensors will not be touched until restored.
2. Multiple processes may run on the same GPU and may contend for the same NUMA/GPU pinned arena.
3. The first priority is to evacuate tensors from GPU VRAM quickly and release GPU memory as soon as the GPU-to-pinned-host copy has completed.

## Contents

```text
spec/DESIGN_SPEC.md              Full architecture and semantics
spec/RACE_CONDITIONS.md          Correctness invariants and races
spec/MEMORY_TIERING.md           Tiering, NUMA, and pinned-drain policy
abi/offload_abi.h                Shared-memory ABI definitions
abi/ABI_NOTES.md                 ABI design notes
rpc/offload_daemon.proto         RPC schema for daemon/rank control plane
rpc/RPC_SEMANTICS.md             Handler semantics and legal transitions
python_api/PYTHON_API.md         User-facing Python interface
impl/IMPLEMENTATION_PLAN.md      Milestones and engineering plan
impl/RANK_AGENT_SKETCH.cpp       Rank-side C++ sketch
impl/DAEMON_SKETCH.cpp           Daemon-side C++ sketch
diagrams/state_machine.mmd       Mermaid state machine diagram
diagrams/architecture.mmd        Mermaid architecture diagram
tests/TEST_PLAN.md               Correctness and performance tests
tests/METRICS.md                 Required metrics and benchmark expectations
```

## Core design in one paragraph

The daemon creates fd-backed, NUMA-local shared-memory arenas and owns slot allocation, leases, tensor-version metadata, background drain to pageable DRAM/NVMe, and prefetch scheduling. Each rank process maps the relevant fd(s), registers its local virtual address range/chunks with `cudaHostRegister()` once, performs GPU↔pinned copies with `cudaMemcpyAsync()`, records CUDA events, and reports completion back to the daemon. GPU VRAM may be released after the D2H event completes; NVMe/pageable writeback is asynchronous and not on the VRAM-release critical path.

## Recommended first implementation

Start with:

1. Single node, single GPU, one pinned arena.
2. Unix domain socket RPC between rank and daemon.
3. Shared-memory control table + fd-backed data arena.
4. Rank-side CUDA event polling and RPC completion reporting.
5. Destructive on-demand Python API: `handle = off.evict(tensor)`.

Then add multi-process safety, pinned drain, NVMe spill, and NUMA-aware per-GPU windows.
