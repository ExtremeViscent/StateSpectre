# Documentation

| Doc | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The whole system — daemon, agent, control plane, tiering, canonical store, rollout pull, and the design decisions behind them |
| [USAGE.md](USAGE.md) | Build, run the daemon, the Python & C++ APIs, configuration, observability, and tests |
| [PERFORMANCE.md](PERFORMANCE.md) | Measured throughput on the target hardware — D2H/H2D, registration, drain/NVMe (O_DIRECT + block sweep), burst, NUMA |

## Design references (`design/`)

Deeper, topic-focused design notes:

| Doc | Topic |
|---|---|
| [design/design-spec.md](design/design-spec.md) | Core architecture & semantics (offload/restore, slots, leases) |
| [design/memory-tiering.md](design/memory-tiering.md) | Pinned → pageable → NVMe tiering and the pinned-drain policy |
| [design/race-conditions.md](design/race-conditions.md) | Correctness invariants and the races they guard against |
| [design/rpc-semantics.md](design/rpc-semantics.md) | Control-plane handler semantics and legal state transitions |
| [design/abi-notes.md](design/abi-notes.md) | Shared-memory ABI design notes |
| [design/canonical-store.md](design/canonical-store.md) | Canonical model-state object model & identity |
| [design/deduplication.md](design/deduplication.md) | Dedup across replicated ranks; semantic vs hash-verified modes |
| [design/rollout-pull-and-export.md](design/rollout-pull-and-export.md) | Sealed manifests, rollout pull flow, export leases |
| [design/multi-job-and-quotas.md](design/multi-job-and-quotas.md) | Job-scoped namespace, per-job accounting & quotas |
| [design/libfabric-backend.md](design/libfabric-backend.md) | The libfabric/`north_comm` export backend + validated RDMA notes |

Schema & config live at the repo root: [`rpc/`](../rpc/) (the `.proto` schema of
record), [`abi/`](../abi/) (shared-memory ABI headers), and [`config/`](../config/)
(example, production, and canonical-store overlay YAML).
