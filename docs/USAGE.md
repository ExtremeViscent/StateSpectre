# Usage & Build Instructions

## Dependencies

Already installed on the target host; listed for portability.

- CUDA 12.9 (`/usr/local/cuda`), GCC 13 (`gcc-toolset-13`), CMake ≥ 3.18
- `numactl-devel` (libnuma) — real `mbind` / NUMA-local arenas
- `liburing` + `liburing-devel` — async O_DIRECT NVMe engine
- PyTorch 2.10 (CXX11 ABI = True) + pybind11 (bundled with torch) — for the Python extension

No protobuf runtime is required or linked (the control plane uses a hand-rolled
binary codec; see [IMPLEMENTATION.md §3](IMPLEMENTATION.md#3-why-a-hand-rolled-wire-protocol-not-protobuf)).

## Build

### C++ (daemon, agent lib, tools, tests, benchmark)

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
```

Produces: `offloadd` (daemon), `offload-stat` (metrics CLI), `offload_bench`
(benchmark), and the `test_*` executables. Libraries: `libofld_common`,
`libofld_daemon`, `libofld_agent`.

### Python extension

Built separately by torch's `cpp_extension` so it matches the libtorch ABI:

```bash
cd python_api
python setup.py build_ext --inplace
python -c "import torch, fastoffload; print('ok')"
```

## Run the daemon

```bash
# Smoke config: one NUMA arena on node 0 for GPU 0.
./build/offloadd --smoke-arena-mb 8192 --numa 0 --gpu 0 --socket /tmp/fastoffload.sock

# Production config (see ../spec/config.production.yaml for all keys):
./build/offloadd --config ../spec/config.production.yaml
```

CLI: `offloadd [--config PATH] [--socket PATH] [--trace PATH] [--smoke-arena-mb N] [--numa N] [--gpu N]`

Environment:
- `OFLD_LOG_LEVEL` — 0=error 1=warn 2=info (default) 3=debug
- `OFLD_TRACE=/path/trace.jsonl` (or `--trace`) — structured JSON trace events

> The daemon uses `memfd_create`; in a restrictive seccomp sandbox that syscall
> may be blocked and the GPU/memfd tests self-skip. Run on the real host.

## Python API

```python
import torch, fastoffload as fo

with fo.offload_context(daemon_addr="unix:///tmp/fastoffload.sock",
                        device="cuda:0", rank=0) as off:
    x = torch.randn(1024, 1024, device="cuda")
    h = off.evict(x, name="x")        # destructive: frees VRAM after D2H completes
    h.wait_offloaded()
    x = h.restore(device="cuda:0")    # byte-identical

    # non-destructive copy (original stays valid)
    hc = off.copy(y)

    # batch
    handles = off.evict_many([k_cache, v_cache], names=["k", "v"])
    off.wait(handles)
    k_cache, v_cache = off.restore_many(handles, device="cuda:0")
```

`off.evict` requires (by default) a CUDA, contiguous, full-storage tensor with
`storage_offset()==0` and no autograd; views/autograd are rejected unless
`allow_views=True, compact=True` or `unsafe_autograd=True`. `handle.restore()`
returns a new tensor. See `../python_api/PYTHON_API.md` for the full surface.

## C++ API (rank agent)

`src/agent/agent.h` — for embedding without Python. The agent owns all CUDA;
device pointers are `uint64_t`, streams are `uintptr_t` (0 = CUDA default
stream, `kInternalStream` = the agent's internal stream):

```cpp
AgentConfig cfg; cfg.socket_path = "/tmp/fastoffload.sock"; cfg.cuda_device = 0;
OffloadAgent agent(cfg);
TensorMeta m{.tensor_id=1, .version=1, .nbytes=nbytes};
agent.evict(dev_ptr, nbytes, /*stream=*/0, /*destructive=*/true, m, cb, cookie, /*wait=*/true);
agent.restore(1, 1, out_dev_ptr, nbytes, /*stream=*/0, /*wait=*/true);
```

## Configuration

`../spec/config.example.yaml` is the baseline; `../spec/config.production.yaml`
documents every tunable added during hardening. Notable knobs:

```yaml
daemon:
  completion_ring_max_ranks: 64      # shared-memory completion rings
  completion_ring_capacity: 1024     # entries per ring (power of two)
nvme:
  io_engine: io_uring                # io_uring | pwrite
  direct_io: true                    # O_DIRECT (recommended, see PERFORMANCE.md)
  block_mb: 16                       # IO block size (16 is the measured sweet spot)
  queue_depth: 16                    # io_uring in-flight blocks
  stripe_dirs: [/nvme0/…, /nvme1/…]  # one dir per drive → stripes span drives
```

## Observability

```bash
./build/offload-stat --socket /tmp/fastoffload.sock            # human table
./build/offload-stat --socket /tmp/fastoffload.sock --prometheus
./build/offload-stat --socket /tmp/fastoffload.sock --watch 2  # refresh every 2s
./build/offload-stat --socket /tmp/fastoffload.sock --shutdown # ask daemon to exit
```

Metrics cover bytes per tier, in-flight/draining bytes, D2H/H2D bandwidth
percentiles, and correctness-rejection counters (stale version, epoch mismatch,
slot-overlap prevented, etc.). Trace events (`OFLD_TRACE`) follow the schema in
`../tests/METRICS.md`.

## Tests

```bash
./run_tests.sh            # build + C++ ctest + Python (e2e, multiproc, LLM)
./run_tests.sh --cpp      # C++ only
./run_tests.sh --python   # Python only (daemon binary must be built)
```

Or directly:

```bash
cd build && ctest --output-on-failure          # 6 C++ suites
# individual: ./test_wire ./test_config ./test_layout ./test_daemon_rpc \
#             ./test_fault_injection ./test_evict_restore
```

Benchmarks: see [PERFORMANCE.md](PERFORMANCE.md#reproducing).

## Repository layout

```
abi/      offload_abi.h — shared-memory ABI (given)
spec/     design docs + config.example.yaml + config.production.yaml
rpc/      offload_daemon.proto (schema of record) + semantics
python_api/  PYTHON_API.md + fastoffload package + setup.py
impl/     original design sketches + IMPLEMENTATION_PLAN.md
src/      the implementation (common / daemon / agent / python / tools)
tests/    cpp + python tests, TEST_PLAN.md, METRICS.md, BENCHMARKS.md
docs/     this documentation
CMakeLists.txt, run_tests.sh
```
