# Memory Tiering, NUMA, and Drain Policy

## 1. Tiers

```text
GPU VRAM
  active tensors

Pinned shared memory
  fast DMA landing zone

Pageable host DRAM
  cold cache if system RAM allows

NVMe
  long-cold capacity tier
```

## 2. Why pinned memory must drain

Pinned memory is scarce and non-pageable. If offloaded tensors are not accessed soon, retaining them in pinned memory wastes DMA capacity and blocks future offloads.

Therefore:

```text
D2H completion -> GPU memory releasable -> enqueue pinned drain
```

Drain is not part of the initial VRAM-release critical path.

## 3. Default drain policy

```text
On D2H completion:
  mark slot PINNED_VALID
  immediately enqueue background drain
```

Pinned pressure thresholds:

```text
< 60% used:
  normal drain

60-80% used:
  prioritize old/large slots

80-95% used:
  aggressive drain; prefer full-window-sized slots

> 95% used:
  block new reservations or force synchronous drain
```

## 4. Drain target policy

```text
If host DRAM pressure is low:
  pinned -> pageable DRAM

If reuse distance is long or DRAM pressure is high:
  pinned -> NVMe

If true capacity extension is needed:
  pageable DRAM -> NVMe in background
```

Default target:

```text
pageable_then_nvme
```

## 5. NUMA policy

Use one arena per NUMA node. Each rank registers the local NUMA arena chunks it may use.

Allocation preference:

```text
1. GPU preferred window
2. same-NUMA overflow
3. borrowable same-NUMA peer window
4. drain
5. remote NUMA fallback/block
```

Remote NUMA allocation should be measured and visible in metrics.

## 6. Capacity planning

For each NUMA node:

```text
pinned_capacity >= maximum expected burst D2H volume before drain catches up
```

If N processes on the same NUMA node can each offload an 80 GB shard concurrently, a 320 GB pinned arena may be necessary just to absorb the burst before any drain completes.

## 7. Bandwidth model

Critical path:

```text
T_release_vram = T_D2H(GPU -> pinned)
```

Background:

```text
T_recycle_pinned = T_drain(pinned -> pageable/NVMe)
```

Offload will block if:

```text
incoming D2H burst volume > free pinned capacity + drained capacity during burst
```
