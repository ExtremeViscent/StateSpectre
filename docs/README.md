# Documentation index

This directory is the consolidated documentation for the centralized GPU
tensor-offload runtime — a full C++/CUDA implementation of the design in
`../spec`, `../abi`, `../rpc`, and `../python_api`.

| Doc | What it covers |
|---|---|
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | What was built: architecture, every component, the design decisions and the real bugs found & fixed, milestone status |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured throughput on the target hardware — D2H/H2D, registration, drain/NVMe (O_DIRECT + block-size sweep), burst, NUMA — with interpretation |
| [USAGE.md](USAGE.md) | How to build, run the daemon, use the Python & C++ APIs, configure, observe metrics, and run the test suite |

## One-paragraph summary

The daemon (`offloadd`, CUDA-free) owns fd-backed NUMA-local pinned arenas, slot
allocation, leases, the tensor location table, background drain to
pageable/NVMe, and heartbeat-based crash recovery. Each rank process maps the
arenas, registers them once with `cudaHostRegister`, and performs GPU↔pinned
`cudaMemcpyAsync` copies, reporting completion to the daemon either through a
lock-free shared-memory ring (no syscall) or a batched RPC. GPU VRAM is released
the moment the D2H CUDA event completes (~1.66 s for 80 GB on PCIe Gen5);
migration to the cold tiers is asynchronous background work that never gates the
release. The Python API (`fastoffload`) exposes `off.evict()` / `handle.restore()`
over this, with real destructive storage replacement.

## Status

All seven milestones in `../impl/IMPLEMENTATION_PLAN.md` are implemented and
verified on real hardware (8× H20, 2× NVMe): single-GPU prototype, destructive
Python API, multi-process lease safety, drain to pageable, NVMe spill,
NUMA-aware multi-GPU, and performance/production hardening (batched +
shared-ring completions, tunable NVMe engine, metrics tool, fault-injection
tests, LLM-style workload, benchmark suite).

Regression: 6 C++ ctest suites + Python end-to-end + 4-process concurrency +
LLM-style workload all pass. See [USAGE.md](USAGE.md#tests) to reproduce.
