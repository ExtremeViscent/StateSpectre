# StateSpectre — project instructions

Centralized GPU offload + canonical model-state runtime. One node-level daemon
(`offloadd`, CUDA-free) owns pinned host arenas, NVMe spill, leases, and the
canonical object/manifest store; training ranks attach through the C++
`OffloadAgent` and the `fastoffload` Python package (which owns all CUDA).

This file orients a developer (or agent) working in this repo. Read
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design; this is the
how-to-work-here.

## Repository layout

```
src/common/     wire protocol, UDS transport, control-region layout, completion
                ring, NUMA utils, metrics — shared by daemon and agent
src/daemon/     offloadd: arenas, slot allocator, leases, location table, cold
                store, drain/readback workers, NVMe engine, crash recovery,
                canonical store (canonical*.cpp), export backends
src/agent/      OffloadAgent: the rank-side CUDA owner + synchronous RPC client
src/python/     bindings.cpp — the ONLY translation unit that knows torch::Tensor
src/tools/      offload-stat (metrics CLI)
python_api/     fastoffload package (ergonomic layer) + setup.py + API docs
abi/            offload_abi.h (v1 shared-memory ABI) + offload_canonical_abi.hpp (v2)
rpc/            *.proto — schema of record for the wire protocol (not linked)
config/         config.example.yaml, config.production.yaml, offload_manager_v2.yaml
tests/cpp/      unit + live-daemon integration tests (own tiny harness)
tests/python/   e2e, multiproc, LLM-workload, cross-node libfabric harness
docs/           ARCHITECTURE, USAGE, PERFORMANCE + docs/design/ references
```

## Build & test

```bash
# C++ (daemon, agent lib, tools, tests, benchmark)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"
ctest --output-on-failure                      # all C++ suites

# Python extension (built by torch cpp_extension to match the libtorch ABI)
cd python_api && python setup.py build_ext --inplace

# Full suite: C++ ctest + Python e2e/multiproc/LLM + canonical rollout e2e
./run_tests.sh            # [--cpp | --python]
```

Tests labelled `memfd` need `memfd_create`; the one labelled `gpu`
(`evict_restore`) needs a real GPU. Both self-skip under a restrictive sandbox —
run on the real host. The test harness is `tests/cpp/test_harness.h`
(`CHECK`/`CHECK_EQ`/`RUN`/`summary`), no external framework.

## Non-negotiable invariants

Preserve these; they are the load-bearing design decisions.

1. **The daemon never links CUDA.** All CUDA (streams, events,
   `cudaHostRegister`, D2H/H2D) lives in the rank agent. The daemon owns memory,
   metadata, and lifetime only. Do not add a CUDA dependency to `src/daemon/`.
2. **No protobuf runtime.** `rpc/*.proto` is the schema *of record*, but the
   wire is a hand-rolled fixed-layout little-endian codec (`src/common/wire*.{h,cpp}`)
   mirroring the message structs in `protocol.h` / `protocol_v2.h`. Reason:
   libtorch statically bundles its own protobuf and the agent loads into the
   torch process — a second protobuf risks ODR/symbol clashes.
3. **v2 is strictly additive over v1.** With `manager.enable_canonical_store:
   false` the daemon is exactly v1. Canonical objects do **not** re-implement
   tiering: each is backed by the v1 slot/lease/drain/NVMe machinery through a
   synthetic `tensor_id` (`canonical_tid(object_id) = 1<<63 | object_id`), so
   the rank drives the unchanged `MarkD2HSubmitted → MarkD2HComplete` path.
4. **Rollout engines pull only SEALED versions.** `PullTensor` / `GetManifest`
   reject anything not sealed. Never expose mutable trainer weights to a puller.
5. **`bindings.cpp` is the only file that includes torch.** `agent.h` speaks
   raw `uint64_t` device pointers and `uintptr_t` streams so the agent links
   without leaking CUDA/torch symbols into other TUs.

## Adding or changing a control-plane RPC

The wire is manual, so a message change touches a fixed set of places — do all
of them together:

1. `src/common/protocol.h` (v1) or `protocol_v2.h` (v2): the request/response
   struct, and the `OpCode` enum value.
2. `src/common/wire.cpp` / `wire_v2.cpp`: `encode()` + `decode_*()` (field order
   must mirror the struct), and the `op_name()` case.
3. `src/daemon/daemon.cpp` `dispatch()`: a `case` that decodes → handler →
   encodes the reply.
4. The handler (`src/daemon/*.cpp`) + its declaration in `daemon.h`.
5. If ranks call it: a method on `RpcClient` (`src/agent/rpc_client.*`) and the
   agent/bindings/Python layers.
6. A codec round-trip check in `tests/cpp/test_wire*.cpp`.

**Rebuild-together gotcha:** daemon and agent/Python share the codec. If you
change a message, rebuild **both** the `build/` daemon and the `python_api`
extension. A mismatch shows up at runtime as `WireError: truncated wire
message` (the classic symptom of a stale `offloadd` vs a rebuilt agent).

## libfabric export backend (optional)

Off by default; `PullTensor(libfabric)` falls back to TCP. To build against
`north_comm`:

```bash
cmake .. -DOFLD_WITH_LIBFABRIC=ON -DNORTH_COMM_DIR=/path/to/north_comm
```

Hard-won build/runtime facts (see [docs/design/libfabric-backend.md](docs/design/libfabric-backend.md)):
- Only `export_libfabric.cpp` compiles at `-std=c++20` (north_comm headers use
  `std::ranges`); the rest of the daemon stays C++17.
- `libnorth_comm.so` is a pybind extension → the daemon must link the matching
  libpython (north_comm here is built for CPython **3.12**; override with
  `-DOFLD_PYTHON_LIB=`). The daemon never calls Python; this only satisfies `ld`.
- Runtime `LD_LIBRARY_PATH` needs the built libfabric lib, `/opt/gdrcopy/lib`,
  and a `liburing.so.1` (absent on some worker nodes).
- Send objects as a 1-D **uint8 Tensor** buffer (DLPack-able), NOT a `Bytes`
  buffer — `Bytes` round-trips through a UTF-8 `str` and corrupts binary weights.
- The working RDMA domain on this cluster is `mlx5_bond_1`; verify any NIC with
  north_comm's own `benchmark/cpu_send_recv.py --server-nic X --client-nic X`
  before use. The daemon opens its local NIC via `OFLD_LIBFABRIC_NIC`.

## Conventions

- Match the surrounding style: the codebase favors dense, explanatory comments
  that state *why*, fixed-layout structs, and explicit memory ordering on shared
  atomics. Keep new code at the same altitude.
- Metrics are a fixed enum (`src/common/metrics.h`) surfaced through `GetStats`;
  add counters there and in `metric_name()` together.
- Config keys parse in `src/daemon/config.cpp`; add the field + default in
  `config.h`, the parse case, and a check in `tests/cpp/test_config.cpp`.
- Commit per logical change with a descriptive body; keep the tree green
  (`ctest`) before committing.
