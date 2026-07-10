# Centralized GPU Offloading System Design Spec

## 1. Goal

Build a centralized offloading system that allows multiple PyTorch processes to quickly evacuate model tensors, temporary buffers, or activations from GPU VRAM into host-side storage.

The primary optimization goal is:

```text
Minimize time until GPU VRAM can be released.
```

Therefore, the critical path is only:

```text
GPU VRAM -> pinned host memory
```

Migration from pinned host memory to pageable DRAM or NVMe is background work and must not block initial GPU memory release.

## 2. Core Design Choice

Use design B:

```text
The daemon provides fd-backed shared CPU memory.
The rank process maps/registers the host range and performs GPU<->CPU DMA.
```

Do not make the daemon directly read GPU memory using CUDA IPC in v1.

### Rationale

This avoids:

- CUDA IPC lifetime hazards for PyTorch CUDA allocations.
- Cross-process GPU pointer ownership.
- Daemon-side CUDA contexts and event importing.
- PyTorch caching allocator suballocation exposure.
- Cross-process stream-ordering complexity.

The rank process already owns CUDA streams, PyTorch tensor references, and allocator safety. It should therefore own GPU↔pinned `cudaMemcpyAsync()` calls.

## 3. System Responsibilities

### Offload daemon owns

```text
- fd-backed shared-memory arenas
- NUMA-local arena layout
- slot allocation and leases
- tensor_id/version -> location table
- pinned pressure policy
- background pinned -> pageable/NVMe drain
- cold-tier readback/prefetch
- process heartbeat and lease cleanup
- metrics and tracing
```

### Rank-side agent owns

```text
- mmap of data/control fds
- cudaHostRegister on local mapped ranges/chunks
- CUDA stream correctness
- D2H/H2D cudaMemcpyAsync
- CUDA event recording/querying
- storage invalidation after D2H completion
- restore into new CUDA tensors
- Python/C++ binding to PyTorch
```

## 4. Memory Tiers

```text
Tier 0: GPU VRAM
  Active tensors.

Tier 1: pinned shared-memory arena
  Fast DMA landing zone for GPU<->CPU copies.

Tier 2: pageable host DRAM
  Medium-cold CPU cache. Optional depending on host memory capacity.

Tier 3: NVMe
  Capacity tier for long-cold tensors.
```

Pinned memory should be treated as a fast evacuation/cache tier, not the authoritative long-term store unless the deployment has abundant RAM and a small live offload set.

## 5. NUMA and Arena Layout

Use one fd-backed arena per NUMA node.

Inside each NUMA arena, create logical per-GPU preferred windows plus shared overflow.

Example dual-socket 8-GPU node:

```text
NUMA node 0 arena
  GPU0 preferred window
  GPU1 preferred window
  GPU2 preferred window
  GPU3 preferred window
  shared overflow window

NUMA node 1 arena
  GPU4 preferred window
  GPU5 preferred window
  GPU6 preferred window
  GPU7 preferred window
  shared overflow window
```

Recommended allocation preference:

```text
1. current GPU's preferred window
2. same-NUMA overflow window
3. borrowable same-NUMA GPU window
4. drain old pinned slots
5. remote NUMA fallback or block
```

This gives predictable locality while retaining utilization flexibility.

## 6. Registration Policy

Registration is one-off per process per mapped range/chunk. It is not per copy.

Startup flow:

```text
daemon:
  memfd_create/shm_open
  ftruncate arena
  optionally first-touch / mbind / NUMA policy
  pass fd + metadata to rank

rank:
  mmap(fd)
  cudaHostRegister(local_addr + chunk_offset, chunk_size, flags)
  cache registration for job lifetime or until chunk eviction
```

Recommended chunking:

```text
arena size: tens to hundreds of GB per NUMA node
registration chunk: 256 MB / 512 MB / 1 GB
allocation granularity: 2 MB / 16 MB / 64 MB, depending on tensor sizes
```

Do not register/unregister per tensor or per copy.

## 7. Critical Offload Protocol

### Step 1: reserve pinned slot

```text
rank -> daemon: RequestOffload(tensor_id, version, nbytes, gpu_id, numa_node)
daemon -> rank: lease_id, slot_id, arena_id, arena_offset, capacity
```

State transition:

```text
FREE -> RESERVED_D2H
```

### Step 2: launch D2H

```cpp
cudaMemcpyAsync(slot_addr, tensor_data_ptr, nbytes, cudaMemcpyDeviceToHost, stream);
cudaEventRecord(done_event, stream);
```

State transition:

```text
RESERVED_D2H -> D2H_IN_FLIGHT
```

### Step 3: event completion

Rank-side progress thread queries the event. When complete:

```text
rank -> daemon: MarkD2HComplete(lease_id, tensor_id, version, slot_id)
```

State transition:

```text
D2H_IN_FLIGHT -> PINNED_VALID
```

At this point, the host copy is valid and GPU VRAM may be released/invalidated.

### Step 4: background drain

Daemon enqueues background drain:

```text
PINNED_VALID -> DRAIN_IN_FLIGHT -> COLD_VALID_PAGEABLE or COLD_VALID_NVME -> FREE
```

The drain must not be on the VRAM-release critical path.

## 8. Prefetch/Restore Protocol

To restore a tensor:

1. Daemon locates latest `(tensor_id, version)`.
2. If cold, daemon reads pageable/NVMe data into a pinned slot.
3. Rank launches H2D from pinned slot to a newly allocated CUDA tensor.
4. Rank records H2D completion event.

State path:

```text
COLD_VALID_* -> READBACK_IN_FLIGHT -> PINNED_VALID -> H2D_IN_FLIGHT -> GPU_VALID
```

The default Python API should return a new restored tensor. In-place restore into an old invalidated tensor is possible but should not be v1 default.

## 9. Tensor Adoption Model

The system does not require tensors to be managed at creation time.

It supports on-demand adoption:

```python
handle = off.evict(existing_tensor)
```

This means:

```text
adopt existing PyTorch CUDA tensor
reserve pinned space
copy GPU -> pinned
after event complete, invalidate original tensor storage
return OffloadHandle
```

The user/runtime contract is strict:

```text
After destructive eviction, the original tensor must not be accessed until restored through the handle.
```

For v1, destructive eviction should require:

```text
- CUDA tensor
- contiguous
- storage_offset == 0
- logical nbytes == underlying storage nbytes
- no autograd dependency unless explicitly unsafe
```

Views can be supported in compact-copy mode, but that mode cannot guarantee freeing the base allocation.

## 10. Invalidation Policy

Do not rely on `resize_(0)` alone to free VRAM. Shrinking tensor size may not release underlying storage.

Use a C++ extension that performs storage replacement after D2H completion, using set-storage semantics:

```text
old tensor storage -> empty CUDA storage / empty tensor
old shape/stride -> zero-sized metadata
```

Public naming may be `invalidate="set_empty"` or `invalidate="zero"`, but implementation must replace storage, not only shape.

## 11. Completion Model

v1 recommendation:

```text
Daemon is CUDA-free.
Rank owns CUDA events.
Rank reports completion via RPC.
```

Rank-side event polling avoids CUDA IPC event complexity and daemon GPU contexts.

## 12. Versioning

Every tensor update uses `(tensor_id, version)`. The daemon's location table must only update latest location if the completion's version is still current.

Invariant:

```text
Stale completion must never overwrite newer tensor metadata.
```

## 13. Expected 80 GB Latency Budget

Approximate practical numbers:

```text
GPU -> pinned host:
  PCIe Gen4: ~2.5-3.5 s
  PCIe Gen5: ~1.3-2.0 s

pinned -> pageable DRAM:
  same NUMA: ~0.8-3.0 s

pinned/pageable -> NVMe:
  single NVMe: ~6-20 s
  multi-NVMe stripe: ~2-5 s

NVMe -> pinned:
  single NVMe: ~6-16 s
  multi-NVMe stripe: ~2-5 s

pinned -> GPU:
  PCIe Gen4: ~2.5-3.5 s
  PCIe Gen5: ~1.3-2.0 s
```

Critical path for freeing VRAM:

```text
GPU -> pinned only
```

## 14. Key Design Rules

1. Registration is one-off per process per range/chunk, not per copy.
2. Pinned memory is the DMA landing zone, not necessarily the long-term store.
3. Daemon owns metadata and leases; rank owns CUDA correctness.
4. VRAM can be released after D2H event completion, not after NVMe writeback.
5. Every write into pinned memory requires a daemon-granted lease.
6. Every tensor location update includes a version.
7. Use per-NUMA arenas with per-GPU preferred windows and same-NUMA overflow.
8. NVMe spill is background work.
9. Pinned pressure triggers drain before full exhaustion.
10. No slot may be overwritten while its latest valid copy exists only in that slot.
