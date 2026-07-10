# Test Plan

## 1. Unit tests: ABI and slot table

- Header magic/version validation.
- Arena descriptor parsing.
- Slot offset -> pointer translation.
- State transition table enforcement.
- Rank epoch rejection.
- Lease owner validation.
- Version update ordering.

## 2. Single-GPU functional tests

### Evict/restore byte correctness

```text
create CUDA tensor with deterministic data
evict destructively
restore
compare bytes/values
```

### Invalidation behavior

```text
evict tensor
wait D2H completion
assert original tensor is zero-sized/empty according to chosen invalidation mode
assert restored tensor matches original
```

### Repeated reuse

```text
repeat evict/restore 1000 times using same registered arena
assert no per-copy cudaHostRegister
assert no slot leaks
```

## 3. Race tests

### Double writer

Two processes request offload concurrently. Assert no two active leases receive overlapping slot ranges.

### Stale version

```text
offload tensor version 10 but delay completion
produce/offload version 11
complete version 10 late
assert latest remains version 11
```

### Rank crash

```text
rank requests lease
starts D2H or stalls
process dies
daemon detects heartbeat failure
lease invalidated
slot recovered or marked error
old epoch completion rejected
```

### Drain race

Force daemon drain worker to attempt to drain before D2H completion. Assert state machine prevents it.

### Reuse race

Force a slot into DRAIN_IN_FLIGHT and request allocation. Assert allocator does not return that slot.

## 4. View and alias tests

### Full-storage tensor accepted

```text
x = torch.empty(N, device='cuda').contiguous()
off.evict(x) succeeds
```

### View rejected by default

```text
base = torch.empty(N, device='cuda')
x = base[1:]
off.evict(x) raises unless allow_views=True
```

### Compact view mode

```text
off.evict(x, allow_views=True, compact=True)
restore returns compact tensor with same logical values
no VRAM-free guarantee is asserted
```

## 5. Autograd tests

Default destructive eviction should reject tensors requiring autograd unless unsafe mode or saved_tensors_hooks integration is used.

## 6. Performance tests

### D2H/H2D bandwidth

Measure:

```text
1 GB, 8 GB, 40 GB, 80 GB tensors
D2H to pinned arena
H2D from pinned arena
same NUMA vs remote NUMA
```

### Registration overhead

Measure:

```text
cudaHostRegister per chunk size: 256 MB, 512 MB, 1 GB, 4 GB
startup eager registration
lazy registration first-use latency
```

### Drain bandwidth

Measure:

```text
pinned -> pageable memcpy
pinned/pageable -> NVMe sequential write
NVMe -> pinned readback
```

### Burst absorption

Simulate N ranks each offloading 80 GB. Measure:

```text
reservation blocking
D2H start latency
pinned pressure
background drain catch-up time
```

## 7. NUMA tests

- GPU local arena bandwidth.
- GPU remote arena bandwidth.
- Same-NUMA overflow behavior.
- Remote-NUMA fallback counter.

## 8. Soak tests

Run with random sizes, random offload/restore order, and process failures for hours.

Assertions:

```text
no metadata corruption
no leaked leases
no stale latest-version location
no slot overlap
no data mismatch after restore
```
