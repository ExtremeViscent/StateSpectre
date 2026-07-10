# Implementation Plan

## Milestone 1: Single-process single-GPU prototype

Scope:

```text
one daemon
one rank
one pinned arena
no pageable/NVMe drain
explicit Python evict/restore
```

Tasks:

1. Daemon creates fd-backed data arena and control table.
2. Rank registers with daemon and receives fds.
3. Rank mmaps arena and calls `cudaHostRegister()` once.
4. Rank can request a slot.
5. Rank launches D2H into slot and H2D restore.
6. Rank event thread reports completion.

Success criteria:

```text
D2H/H2D bandwidth close to expected PCIe limit
no per-copy cudaHostRegister
basic evict/restore produces byte-identical tensor
```

## Milestone 2: Destructive on-demand Python API

Scope:

```text
off.evict(existing_tensor)
handle.restore()
validation for full-storage contiguous tensors
storage invalidation after D2H completion
```

Tasks:

1. C++ extension captures tensor metadata.
2. C++ extension validates storage ownership.
3. Event thread invokes storage replacement after D2H completion.
4. Handle stores shape/stride/dtype/device metadata.
5. Restore returns a new CUDA tensor.

Success criteria:

```text
original tensor becomes zero-sized/empty after event completion
GPU memory becomes reusable by process
restored tensor matches original contents
unsafe cases are rejected by default
```

## Milestone 3: Multi-process lease safety

Scope:

```text
multiple ranks/processes on same GPU/NUMA arena
lease ownership
rank_epoch
state checks
```

Tasks:

1. Implement lease table.
2. Add rank heartbeat and epoch invalidation.
3. Reject stale completions.
4. Add blocked allocation/backpressure path.
5. Add stress tests with concurrent writers.

Success criteria:

```text
no two processes write same slot
rank death does not corrupt metadata
stale completion cannot overwrite latest version
```

## Milestone 4: Background drain to pageable DRAM

Scope:

```text
PINNED_VALID -> DRAIN_IN_FLIGHT -> COLD_VALID_PAGEABLE -> FREE
```

Tasks:

1. Add pageable memory cache.
2. Drain workers copy pinned to pageable.
3. Pinned slot recycled after drain.
4. Prefetch copies pageable back into pinned.
5. Restore from pageable path.

Success criteria:

```text
VRAM release latency unaffected by drain
pinned slots recycle under pressure
restore from pageable succeeds
```

## Milestone 5: NVMe spill

Scope:

```text
cold-tier NVMe writeback/readback
```

Tasks:

1. NVMe allocator: file offsets or block-offset abstraction.
2. Async sequential write workers.
3. Async readback workers.
4. Checksums optional.
5. Prefetch from NVMe.

Success criteria:

```text
NVMe is never on initial VRAM-release critical path
large sequential writes/reads hit expected bandwidth
restore from NVMe succeeds
```

## Milestone 6: NUMA-aware multi-GPU

Scope:

```text
one arena per NUMA node
per-GPU preferred windows
overflow windows
remote-NUMA fallback metrics
```

Tasks:

1. Discover GPU<->NUMA affinity.
2. Allocate/bind arena pages to NUMA node.
3. Per-GPU window allocator.
4. Same-NUMA overflow allocator.
5. Remote-NUMA fallback/backpressure.

Success criteria:

```text
local NUMA bandwidth stable
remote NUMA usage visible
uneven offload pressure handled by overflow
```

## Milestone 7: Performance and production hardening

Tasks:

1. Batch evict/restore.
2. Batch completion RPC.
3. Shared-memory completion rings if RPC overhead matters.
4. Trace export.
5. Metrics server.
6. Failure injection.
7. Integration with representative LLM workload.
