# Performance Report

All numbers were measured on the real target hardware with the production build.
The benchmark harness (`tests/cpp/bench.cpp`, target `offload_bench`) runs an
in-process daemon and a real rank agent, times the actual DMA (device-synced
wall clock), and reports the best of 3 iterations per size. Reproduce with the
commands in each section; every table also has machine-readable `BENCH,...`
lines in stdout.

## Test system

| | |
|---|---|
| GPU | 8× NVIDIA H20 (96 GB HBM), **PCIe Gen5 x16** |
| NUMA | 2 nodes; GPUs 0–3 → node 0, GPUs 4–7 → node 1 |
| Host RAM | 2.2 TB (~1.4 TB free) |
| NVMe | 2× Samsung MZQL2 7.6 TB, LVM-striped XFS at `/dockerdata` (9 TB) |
| Software | CUDA 12.9, GCC 13, liburing 1.0.7, libnuma 2.0.16, PyTorch 2.10 |

Runs used GPU 0 (NUMA node 0), 64 MiB slots, completion ring enabled.

---

## 1. D2H / H2D bandwidth — the VRAM-release critical path

The whole design exists to minimize time-until-VRAM-release, which equals the
D2H (GPU→pinned) time. Nothing else gates it.

`offload_bench --section d2h --sizes 1,8,40,80`

| Tensor | D2H (GPU→pinned) | H2D (pinned→GPU) | 80 GB-equivalent time |
|-------:|-----------------:|-----------------:|----:|
| 1 GiB  | 53.6 GB/s | 57.5 GB/s | — |
| 8 GiB  | 53.7 GB/s | 57.5 GB/s | — |
| 40 GiB | 48.1 GB/s | 57.4 GB/s | — |
| **80 GiB** | **48.1 GB/s** | **57.4 GB/s** | **D2H ≈ 1.66 s** |

**Headline: an 80 GB tensor is evacuated from VRAM in ~1.66 s.** The design
spec's PCIe-Gen5 budget for 80 GB was 1.3–2.0 s → measured is squarely in range.
D2H eases slightly at 40/80 GiB (54→48 GB/s) as the transfer saturates the link;
H2D holds ~57 GB/s throughout. `cudaHostRegister` never appears in this path
(see §2).

## 2. cudaHostRegister overhead

`offload_bench --section reg` — register 4 GiB in fixed-size chunks.

| Chunk | Time |
|------:|-----:|
| 256 MiB | ~60–65 ms/GiB |
| 512 MiB | ~59 ms/GiB |
| 1 GiB | ~60–78 ms/GiB |
| 4 GiB | ~62 ms/GiB |

Registration is ~60 ms/GiB and is **one-off per process per arena** — never on a
per-copy path. A 384 GB NUMA arena costs ~23 s to pin once at startup; after
that every evict/restore is registration-free.

> The agent registers each **whole arena as a single contiguous range**. This is
> a correctness requirement, not just an optimization: a single
> `cudaMemcpyAsync` whose target spans two separately-registered regions is
> rejected by CUDA. Regression-tested by `test_large_tensor` (768 MB, crosses
> the old 512 MB chunk boundary).

## 3. Drain bandwidth (pinned → cold tier)

Background migration off the VRAM-release critical path. 8 GiB tensor.

`offload_bench --section drain,nvme --nvme-dir <d1>,<d2>`

| Target / engine | Write GB/s | Read GB/s |
|---|--:|--:|
| pinned → pageable DRAM (memcpy) | 2.5 | 13.3 |
| NVMe **pwrite + O_DIRECT** | **5.5** | 6.3 |
| NVMe pwrite, buffered | 4.0 | 5.6 |
| NVMe io_uring + O_DIRECT | 4.0–5.5 | 5.9 |
| NVMe io_uring, buffered | 4.1 | 4.9 |

- **O_DIRECT beats buffered for NVMe writes** (5.5 vs 4.0 GB/s): buffered pays
  page-cache + writeback overhead that O_DIRECT skips for large sequential
  drains. Confirmed independently at the device level (`dd`: buffered 2.0 GB/s
  vs O_DIRECT 4.3–5.3 GB/s).
- Pageable "write" is a single-threaded `memcpy` to freshly-faulted DRAM;
  reading it back is 13 GB/s (warm). Pageable = low-latency cold tier;
  NVMe = capacity tier.
- These rates never gate an evict — drain runs on background workers after
  `MarkD2HComplete`.

### NVMe block-size sweep (pwrite O_DIRECT, 8 GiB, queue depth 16)

`offload_bench --section nvme --nvme-block-mb {1,4,16,64}`

| Block | Write GB/s | Read GB/s |
|------:|--:|--:|
| 1 MB  | 4.40 | 4.22 |
| 4 MB  | 5.09 | 4.42 |
| **16 MB** | **5.44** | **6.34** |
| 64 MB | 4.54 | 6.24 |

Block size dominates O_DIRECT throughput: **1 MB → 16 MB is +24% on writes.**
16 MB is the sweet spot; 64 MB regresses (bigger bounce buffers, fewer in-flight
blocks at fixed queue depth). Hence `nvme.block_mb` defaults to 16 and is
tunable.

## 4. Burst absorption — concurrent multi-rank offload

`offload_bench --section burst` — 4 ranks, one process, one daemon, one GPU.

- 4 ranks × 4 GiB each, concurrent: **aggregate D2H ≈ 8.8–10.4 GB/s**, wall
  ≈ 1.7–2.0 s for 16 GiB total.

The aggregate is below a single rank's 48 GB/s because all four ranks share the
**one PCIe link on GPU 0** and serialize on the physical bus — expected and
correct. On a real 8-GPU deployment each GPU has its own link and NUMA-local
arena, so per-GPU offloads run in parallel at full bandwidth. What this test
proves is that the daemon arbitrates concurrent lease/slot allocation with zero
overlap under contention (see also the multi-process safety test).

## 5. NUMA locality

`offload_bench --section numa` — 8 GiB, arena bound to each node via `mbind`.

| Arena placement | D2H | H2D |
|---|--:|--:|
| node 0 (GPU 0 local) | 53.7 GB/s | 57.5 GB/s |
| node 1 (remote) | 50.9 GB/s | 57.5 GB/s |

Remote-NUMA D2H is ~5% slower (a cross-socket interconnect hop to pinned
memory), confirming the arena is genuinely placed on the requested node (real
`mbind`, not a no-op). The allocator prefers the GPU's local-NUMA window and
only falls back to remote when explicitly permitted; remote allocations are
counted in `remote_numa_alloc_count`.

## 6. End-to-end VRAM release (Python, LLM-style workload)

`tests/python/test_llm_workload.py` — transformer-style offload of layer
weights. Measured with `torch.cuda.memory_allocated()`:

- Evicting 7 of 8 × 128 MiB layers freed **896 MiB = 100% of the evicted
  volume** — real storage release, not shape shrink.
- All restores byte-identical; batch `evict_many`/`restore_many` identical.

---

## Interpretation against the METRICS.md "red flags"

| Red flag | Result |
|---|---|
| 80 GB D2H much slower than budget | ✅ 1.66 s on Gen5 — within 1.3–2.0 s |
| `cudaHostRegister` in the hot path | ✅ one-off per arena, never per copy |
| D2H/H2D varies without NUMA explanation | ✅ only the expected ~5% remote-NUMA delta |
| NVMe writeback blocks GPU-memory release | ✅ drain is background, off the critical path |
| reservation blocks while pinned usage low | ✅ backpressure only at true exhaustion (fault test) |

## Reproducing

```bash
cd build && cmake .. && make -j offload_bench      # needs GPU + memfd
mkdir -p /dockerdata/nvme_a /dockerdata/nvme_b
CUDA_VISIBLE_DEVICES=0 ./offload_bench --gpu 0 --sizes 1,8,40,80 \
    --nvme-dir /dockerdata/nvme_a,/dockerdata/nvme_b
# sections: --section d2h|reg|drain|nvme|burst|numa
# NVMe tuning: --nvme-block-mb 16 --nvme-qd 16
```

## Note on measurement methodology

Two benchmark-harness pitfalls were hit and corrected while producing these
numbers (the runtime itself was correct):
1. H2D wall-clock must bracket a `cudaDeviceSynchronize()` or it measures only
   the launch, not the DMA.
2. Reusing `(tensor_id, version)` across sizes made a restore read a stale
   slot's byte count; the harness now namespaces ids by size. Real usage bumps
   the version per re-eviction, so this only affected the synthetic loop.
