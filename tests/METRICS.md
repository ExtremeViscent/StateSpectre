# Metrics and Benchmark Expectations

## 1. Required metrics

### Per-rank

```text
rank_id
gpu_id
numa_node
registered_pinned_bytes
used_pinned_bytes
inflight_d2h_bytes
inflight_h2d_bytes
draining_bytes
blocked_reservation_count
remote_numa_alloc_count
```

### Transfers

```text
D2H bytes
D2H duration
D2H bandwidth
H2D bytes
H2D duration
H2D bandwidth
pinned_to_pageable bytes/duration/bandwidth
pageable_to_pinned bytes/duration/bandwidth
NVMe write bytes/duration/bandwidth
NVMe read bytes/duration/bandwidth
```

### Latency stages

```text
offload request -> lease granted
lease granted -> D2H submitted
D2H submitted -> D2H complete
D2H complete -> original tensor invalidated
PINNED_VALID -> drain start
drain start -> cold valid
prefetch request -> pinned ready
pinned ready -> H2D complete
```

### Correctness counters

```text
invalid transition rejected
stale version completion rejected
rank epoch mismatch rejected
slot overlap prevented
view rejected
autograd unsafe rejected
checksum mismatch
IO failure
```

## 2. Expected 80 GB latency budget

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

## 3. Red flags

Investigate if:

```text
80 GB D2H on PCIe Gen4 is much slower than ~4-5 s
D2H/H2D bandwidth varies heavily without NUMA explanation
cudaHostRegister appears in hot path
reservation blocks while pinned usage is low
remote-NUMA allocations happen frequently
NVMe writeback blocks GPU memory release
slots remain D2H_IN_FLIGHT after rank event completion
```

## 4. Trace events

Emit structured trace events:

```json
{
  "ts_ns": 0,
  "rank": 0,
  "gpu": 0,
  "tensor_id": 123,
  "version": 4,
  "lease_id": 88,
  "slot_id": 9,
  "event": "D2H_COMPLETE",
  "nbytes": 85899345920
}
```

Recommended event names:

```text
REQUEST_OFFLOAD
LEASE_GRANTED
D2H_SUBMITTED
D2H_COMPLETE
TENSOR_INVALIDATED
DRAIN_ENQUEUED
DRAIN_STARTED
COLD_VALID
PREFETCH_REQUESTED
READBACK_STARTED
PINNED_READY
H2D_SUBMITTED
H2D_COMPLETE
LEASE_RELEASED
ERROR
```
