# Implementation Summary

A complete, tested C++/CUDA implementation of the centralized GPU tensor-offload
runtime. ~7.3k lines of implementation + ~2.5k lines of tests. No placeholders;
every path is real and exercised.

## 1. Architecture

```
                       ┌─────────────────────────────────────────────┐
   rank process 0      │              offloadd  (CUDA-free)           │
 ┌──────────────────┐  │  ┌────────────────────────────────────────┐ │
 │ Python fastoffload│ │  │ per-NUMA pinned arenas (memfd + mbind)   │ │
 │   off.evict/…     │ │  │ slot allocator (per-GPU windows+overflow)│ │
 │  ↕ (pybind11)     │ │  │ lease / location / session tables        │ │
 │ OffloadAgent (C++)│ │  │ drain + readback worker pools            │ │
 │  cudaHostRegister │ │  │ NVMe engine (io_uring / pwrite, O_DIRECT) │ │
 │  D2H/H2D + events │ │  │ heartbeat monitor + lease recovery       │ │
 │  progress thread  │ │  │ completion-ring poller                   │ │
 └───────┬──────────┘  │  │ metrics + JSON trace                     │ │
         │             │  └────────────────────────────────────────┘ │
         │ AF_UNIX (control plane, SCM_RIGHTS fd passing)             │
         │ + shared control region (slot table + completion rings)   │
         └────────────────────────────────────────────────────────────┘
              rank 1 … N attach the same fds and control region
```

**Division of responsibility (design choice "B"):** the rank owns all CUDA
(streams, events, `cudaHostRegister`, the actual copies) because it already
owns the PyTorch allocator and stream state; the daemon owns metadata, memory,
and lifetime. The daemon never links CUDA. This avoids CUDA-IPC lifetime
hazards, cross-process GPU pointer ownership, and daemon-side CUDA contexts.

## 2. Components

| Path | Role |
|---|---|
| `src/common/protocol.h`, `wire.{h,cpp}` | RPC message structs + fixed-layout little-endian binary codec |
| `src/common/uds.{h,cpp}` | AF_UNIX transport, length-framed, `SCM_RIGHTS` fd passing |
| `src/common/control_layout.h` | Layout of the shared control region (header, arena descs, slot table, ring area) |
| `src/common/completion_ring.h` | Lock-free SPSC completion ring (shared-memory hot path) |
| `src/common/numa_util.{h,cpp}` | `mbind`/first-touch, GPU→NUMA affinity via sysfs, thread pinning |
| `src/common/metrics.{h,cpp}` | Atomic counters, bandwidth percentiles, JSON trace |
| `src/daemon/daemon.{h,cpp}` | Arenas, allocator, tables, all RPC handlers, workers, recovery, NVMe |
| `src/daemon/config.{h,cpp}` | Hand-written YAML-subset config parser |
| `src/daemon/main.cpp` | `offloadd` CLI entry point |
| `src/agent/agent.{h,cpp}` | `OffloadAgent`: the rank-side C++ API (CUDA lives here) |
| `src/agent/rpc_client.{h,cpp}` | Synchronous RPC client over the UDS transport |
| `src/python/bindings.cpp` | pybind11 `_fastoffload` + torch storage invalidation |
| `python_api/fastoffload/` | Ergonomic Python API (`offload_context`, `OffloadHandle`, …) |
| `src/tools/offload_stat.cpp` | Metrics CLI (human / Prometheus / watch / shutdown) |

## 3. Why a hand-rolled wire protocol (not protobuf)

libtorch statically bundles its own protobuf, and the rank agent is loaded into
the PyTorch process. Linking a second protobuf runtime risks ODR/symbol clashes
that surface as random crashes. Since both ends of the socket are ours and the
schema is fixed, `src/common/wire.cpp` is a compact codec that mirrors
`rpc/offload_daemon.proto` field-for-field. `protoc` remains available as the
schema of record; no protobuf runtime is linked anywhere.

## 4. The offload / restore protocol

**Evict (destructive):**
1. `RequestOffload` → daemon reserves a contiguous slot run, `FREE → RESERVED_D2H`, returns arena offset (never a pointer).
2. Rank ensures the arena is registered (once), `cudaMemcpyAsync` D2H, records an event.
3. Progress thread sees the event complete → reports `MarkD2HComplete` (ring or RPC), `D2H_IN_FLIGHT → PINNED_VALID`. **Only now** is the host copy valid and the original GPU storage replaced with an empty tensor (real set-storage, not `resize_(0)`).
4. Daemon enqueues background drain: `PINNED_VALID → DRAIN_IN_FLIGHT → COLD_VALID_{PAGEABLE,NVME} → FREE`.

**Restore:** `RequestPrefetch` → if cold, daemon allocates a pinned slot and starts readback (`COLD → READBACK_IN_FLIGHT → PINNED_VALID`); rank polls readiness, then H2D into a fresh CUDA tensor.

## 5. Correctness invariants (all covered by tests)

- VRAM freed only after the D2H **event** completes (RACE §1).
- Drain can only start after `MarkD2HComplete`; `DRAIN_IN_FLIGHT` slots aren't allocatable (RACE §2/§3).
- Every pinned write requires a daemon lease; concurrent writers never share a slot (RACE §4).
- Location updates are conditional on `version ≥ current` — stale completions recorded but never overwrite the latest (RACE §5).
- Every RPC/ring entry carries `rank_epoch`; old-process completions after restart are rejected (RACE §6).
- Destructive evict validates full-storage ownership (contiguous, offset 0, logical == storage bytes) and rejects autograd/views by default (RACE §7/§8).
- A slot holding the only latest copy is never freed/overwritten (RACE §9).
- H2D never reads a `READBACK_IN_FLIGHT` slot (RACE §10).
- Heartbeat timeout recovers a dead rank's in-flight slots; completed (`PINNED_VALID`) data survives owner death.

## 6. Performance features (M7)

- **Shared-memory completion ring** (`completion_ring.h`): one lock-free SPSC
  ring per rank in the control region. The rank publishes completions with no
  socket syscall; the daemon's poller applies them. Ring metadata lives in the
  slot-table header's reserved bytes (ABI-compatible). Falls back to a single
  `BatchComplete` RPC when the ring is full or unassigned.
- **Batched completions:** all completions gathered in one progress-thread poll
  coalesce into one RPC when the ring path isn't used.
- **Production NVMe engine:** real `io_uring` async O_DIRECT (and a `pwrite`
  engine), **block-chunked with a bounded bounce-buffer pool** — an 80 GB tensor
  never allocates more than `queue_depth × block_bytes` of staging. Tunable
  `block_mb`, `queue_depth`, and per-drive `stripe_dirs` for real multi-drive
  striping.

## 7. Real bugs found and fixed during hardening

These are worth recording because they were subtle and are now regression-tested:

1. **Registration boundary bug (serious).** A single `cudaMemcpyAsync` whose
   target spans two separately-`cudaHostRegister`'d regions is rejected by CUDA
   ("invalid argument"). The original chunked registration therefore corrupted
   /failed any tensor larger than the registration chunk. Fixed by registering
   **each whole arena as one contiguous range**. Regression: `test_large_tensor`
   evicts a 768 MB tensor crossing the old 512 MB boundary; 1/2/4 GB verified
   byte-identical.
2. **Stream-sentinel bug.** `StreamHandle == 0` had meant "use the agent's
   internal stream", but 0 is also the real CUDA default stream — so a D2H ran
   unordered w.r.t. the producer kernel and lost the tail under multi-process
   GPU contention. Fixed: 0 is the default stream; `kInternalStream` is a
   distinct sentinel.
3. **Prefetch slot leak.** Polling `RequestPrefetch` allocated a fresh readback
   run every call. Made the daemon prefetch idempotent per `(tensor_id,version)`.
4. **Cross-process tensor_id collision.** Auto-assigned ids now namespace by
   rank so concurrent processes never collide in the global location table.
5. **Registration leak on teardown.** The agent munmap'd arenas without
   `cudaHostUnregister`, causing "resource already mapped" on re-map. Fixed.

## 8. Milestone status (`../impl/IMPLEMENTATION_PLAN.md`)

| M | Scope | Status |
|---|---|---|
| 1 | Single-GPU prototype | ✅ |
| 2 | Destructive Python API | ✅ |
| 3 | Multi-process lease safety | ✅ |
| 4 | Drain → pageable DRAM | ✅ |
| 5 | NVMe spill (io_uring + pwrite) | ✅ |
| 6 | NUMA-aware multi-GPU | ✅ |
| 7 | Perf / production hardening | ✅ (batch + ring completions, tunable NVMe, metrics tool, fault tests, LLM workload, benchmarks) |

## 9. Test inventory

- **C++ unit:** `test_wire` (codec), `test_config` (YAML parser), `test_layout` (control-region math).
- **C++ integration (live daemon, no GPU):** `test_daemon_rpc` (state machine + races incl. concurrent double-writer), `test_fault_injection` (crash recovery, stale epoch, exhaustion backpressure, idempotent release).
- **C++ functional (GPU):** `test_evict_restore` (byte-correctness, 200× reuse, non-destructive copy, large-tensor boundary crossing, drain→pageable, NVMe both engines).
- **Python:** `test_python_e2e` (full API surface), `test_multiproc` (N processes, no corruption/overlap), `test_llm_workload` (transformer-style offload with real VRAM-release measurement).
- **Benchmark:** `offload_bench` (see [PERFORMANCE.md](PERFORMANCE.md)).
