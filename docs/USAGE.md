# Usage

## Dependencies

- CUDA 12.x (`/usr/local/cuda`), GCC 13, CMake ≥ 3.18
- `libnuma` (`numactl-devel`) — real `mbind` / NUMA-local arenas
- `liburing` (`liburing-devel`) — async O_DIRECT NVMe engine
- PyTorch (CXX11 ABI) + pybind11 (bundled with torch) — for the Python extension

No protobuf runtime is linked; the control plane uses a hand-rolled binary codec
(see [ARCHITECTURE.md](ARCHITECTURE.md#control-plane)). The libfabric export
backend is optional and off by default.

## Build

```bash
# C++: daemon, agent lib, tools, tests, benchmark
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
```

Produces `offloadd` (daemon), `offload-stat` (metrics CLI), `offload_bench`
(benchmark), the `test_*` executables, and the `libofld_*` libraries.

```bash
# Python extension — built by torch's cpp_extension to match the libtorch ABI
cd python_api
python setup.py build_ext --inplace
python -c "import torch, state_spectre; print('ok')"
```

Optional libfabric/RDMA export backend:

```bash
cmake .. -DOFLD_WITH_LIBFABRIC=ON -DNORTH_COMM_DIR=/path/to/north_comm
```

## Run the daemon

```bash
# Smoke config: one NUMA arena on node 0 for GPU 0.
./build/offloadd --smoke-arena-mb 8192 --numa 0 --gpu 0 --socket /tmp/state_spectre.sock

# Production config (all keys documented in config/config.production.yaml):
./build/offloadd --config config/config.production.yaml

# With a TCP control port for remote rollout engines:
./build/offloadd --config config/config.production.yaml --tcp-port 19090
```

CLI: `offloadd [--config PATH] [--socket PATH] [--trace PATH] [--tcp-port N]
[--smoke-arena-mb N] [--numa N] [--gpu N]`

Environment:
- `OFLD_LOG_LEVEL` — 0=error 1=warn 2=info (default) 3=debug
- `OFLD_TRACE=/path/trace.jsonl` (or `--trace`) — structured JSON trace events

> The daemon uses `memfd_create`; under a restrictive seccomp sandbox that
> syscall may be blocked and the memfd/GPU tests self-skip. Run on the real host.

## Python API — tensor offload

```python
import torch, state_spectre as fo

with ss.offload_context(daemon_addr="unix:///tmp/state_spectre.sock",
                        device="cuda:0", rank=0) as off:
    x = torch.randn(1024, 1024, device="cuda")
    h = off.evict(x, name="x")        # destructive: frees VRAM after D2H completes
    h.wait_offloaded()
    x = h.restore(device="cuda:0")    # byte-identical

    hc = off.copy(y)                  # non-destructive: original stays valid

    handles = off.evict_many([k_cache, v_cache], names=["k", "v"])
    off.wait(handles)
    k_cache, v_cache = off.restore_many(handles, device="cuda:0")
```

`off.evict` requires (by default) a CUDA, contiguous, full-storage tensor with
`storage_offset()==0` and no autograd; views/autograd are rejected unless
`allow_views=True, compact=True` or `unsafe_autograd=True`. Full surface:
[python_api/PYTHON_API.md](../python_api/PYTHON_API.md).

## Python API — canonical model-state (RL post-training)

```python
import state_spectre as ss

# Trainer: job-aware context, canonical evict, seal + promote a rollout version.
with ss.offload_context(daemon_addr="unix:///tmp/state_spectre.sock", device="cuda:0",
                        job_name="qwen32b_grpo", scheduler_job_id=SLURM_JOB_ID,
                        dedup_policy=ss.DedupPolicy(mode="semantic_trusted")) as off:
    for name, w in named_weights:                      # e.g. HF-named tensors
        key = off.canonical_key(model_role="policy_rollout", model_version=step,
                                param_name=name, tensor=w,
                                pp_rank=pp, tp_rank=tp, ep_rank=ep, expert_id=-1)
        off.canonical_evict(w, key, destructive=True)  # dp excluded → replicas dedup
    off.promote_rollout_version(step)                  # seal + publish
```

```python
# Rollout / inference engine (may be remote), over the daemon's TCP control port.
client = ss.RolloutWeightClient(daemon_addr="tcp://trainer-node:19090",
                                job_id=job_id, launch_epoch=launch_epoch,
                                model_role="policy_rollout")
version  = client.get_latest_sealed_version()
manifest = client.get_manifest(version)
for entry in client.diff_local(manifest):              # only changed objects
    raw = client.pull_tensor(entry, version, transport="tcp")   # or "libfabric"
    load_into_engine(entry["key"]["param_fqn_debug"], raw)
```

Full surface: [python_api/canonical_python_api.md](../python_api/canonical_python_api.md).

## C++ API (rank agent)

`src/agent/agent.h` — for embedding without Python. The agent owns all CUDA;
device pointers are `uint64_t`, streams are `uintptr_t` (0 = CUDA default
stream, `kInternalStream` = the agent's internal stream):

```cpp
AgentConfig cfg; cfg.socket_path = "/tmp/state_spectre.sock"; cfg.cuda_device = 0;
OffloadAgent agent(cfg);
TensorMeta m{.tensor_id=1, .version=1, .nbytes=nbytes};
agent.evict(dev_ptr, nbytes, /*stream=*/0, /*destructive=*/true, m, cb, cookie, /*wait=*/true);
agent.restore(1, 1, out_dev_ptr, nbytes, /*stream=*/0, /*wait=*/true);
```

The canonical/rollout RPCs are also on `OffloadAgent` (`register_job`,
`canonical_evict`, `seal_model_version`, `get_manifest`, `pull_tensor`).

## Configuration

`config/config.example.yaml` is the baseline and `config/config.production.yaml`
documents every tunable; `config/offload_manager_v2.yaml` is the canonical-store
overlay (job namespace, dedup policy, manifest retention, export transport,
per-job quotas). Notable knobs:

```yaml
daemon:
  completion_ring_max_ranks: 64      # shared-memory completion rings
  completion_ring_capacity: 1024     # entries per ring (power of two)
nvme:
  io_engine: io_uring                # io_uring | pwrite
  direct_io: true                    # O_DIRECT (recommended — see PERFORMANCE.md)
  block_mb: 16                       # IO block size (16 is the measured sweet spot)
  queue_depth: 16                    # io_uring in-flight blocks
  stripe_dirs: [/nvme0/…, /nvme1/…]  # one dir per drive → stripes span drives
manager:                             # v2 canonical store
  enable_canonical_store: true
  control_tcp_port: 19090
canonical_store:
  dedup_default_mode: semantic_trusted   # disabled | semantic_trusted | hash_verified | sampled_hash
export:
  default_transport: libfabric_send      # libfabric_send | tcp | file
  fallback_transport: tcp
quotas:
  default: { max_pinned_bytes: 256GiB, max_nvme_bytes: 8TiB }
```

## Observability

```bash
./build/offload-stat --socket /tmp/state_spectre.sock            # human table
./build/offload-stat --socket /tmp/state_spectre.sock --prometheus
./build/offload-stat --socket /tmp/state_spectre.sock --watch 2  # refresh every 2s
./build/offload-stat --socket /tmp/state_spectre.sock --shutdown # ask daemon to exit
```

Metrics cover bytes per tier, in-flight/draining bytes, D2H/H2D bandwidth
percentiles, canonical dedup savings, manifest/rollout counters, and
correctness-rejection counters. Trace events (`OFLD_TRACE`) follow the schema in
[../tests/METRICS.md](../tests/METRICS.md).

## Tests

```bash
./run_tests.sh            # build + C++ ctest + Python (e2e, multiproc, LLM, canonical rollout)
./run_tests.sh --cpp      # C++ only
./run_tests.sh --python   # Python only (daemon binary must be built)

cd build && ctest --output-on-failure   # C++ suites directly
```

C++ suites: `wire`, `wire_v2`, `config`, `layout`, `daemon_rpc`,
`fault_injection`, `canonical`, `evict_restore` (GPU). The `daemon_rpc` /
`canonical` suites need `memfd_create`; `evict_restore` needs a GPU. Cross-node
libfabric validation lives in `tests/python/` (see
[design/libfabric-backend.md](design/libfabric-backend.md)). Benchmarks:
[PERFORMANCE.md](PERFORMANCE.md#reproducing).

## Repository layout

```
src/         implementation — common / daemon / agent / python / tools
abi/         shared-memory ABI headers (v1 + canonical v2)
rpc/         *.proto — wire-protocol schema of record (not linked)
config/      example + production + canonical-store overlay YAML
python_api/  state_spectre package, setup.py, API docs
tests/       cpp + python tests, TEST_PLAN(_V2), METRICS, BENCHMARKS
docs/        ARCHITECTURE, USAGE, PERFORMANCE + docs/design references
```
