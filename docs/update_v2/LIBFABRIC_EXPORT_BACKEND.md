# Libfabric Export Backend

## Role

The libfabric sender is a transport backend for canonical objects. It should not know about PyTorch tensors directly. It receives object/export buffer descriptors from the offload daemon.

## Preferred v1 Export Flow

```text
PullTensor(object_id, target_descriptor)
  -> daemon locates canonical object
  -> daemon stages object into dedicated export buffer if needed
  -> daemon creates export lease
  -> libfabric backend sends export buffer
  -> completion releases export lease
```

## Why Dedicated Export Buffers First

Directly exposing pinned offload slots adds lifetime risks:

```text
daemon drain/recycle wants slot
remote rollout is still reading slot
```

Dedicated export buffers decouple the offload cache from remote transport.

## Export Buffer Descriptor

```cpp
struct ExportBufferDesc {
    uint64_t export_id;
    uint64_t object_id;
    void*    local_addr;
    uint64_t nbytes;
    uint64_t content_hash_lo;
    uint64_t content_hash_hi;
    uint32_t numa_node;
    uint32_t flags;
};
```

Shared ABI should not store raw pointers. Internal backend descriptors may use local process pointers.

## Transport Modes

### Request-triggered send

Rollout asks trainer to send.

Pros:

```text
simple lifetime control
trainer controls staging
works with pageable/NVMe source
```

Recommended first mode.

### RDMA read

Rollout reads from trainer-exposed memory.

Pros:

```text
lower trainer CPU involvement
can be zero-copy from pinned export buffer
```

Requires:

```text
registered memory
rkey exposure
strict export lease lifetime
remote access control
```

### RDMA write

Trainer writes into rollout-provided buffer.

Useful if rollout pre-registers destination buffers.

## PullTensor Target Descriptor

Opaque bytes should encode transport-specific information, for example:

```text
endpoint address
remote memory key
remote address
destination tensor id
expected nbytes
```

Keep this opaque at the canonical RPC layer.

## Completion Handling

Backend must report:

```text
export_id
object_id
bytes_sent
status
error_code
duration_ns
```

Daemon then:

```text
export_refcount--
release export buffer
update metrics
```

## Safety Invariants

```text
object/export buffer cannot be freed while export_refcount > 0
rollout can only pull sealed manifests
transport must verify nbytes and object_id
failed export must not corrupt object state
```

## Metrics

```text
libfabric_export_requests_total
libfabric_export_bytes_total
libfabric_export_duration_ns
libfabric_export_errors_total
libfabric_export_retries_total
libfabric_export_buffer_wait_ns
```

---

## Implementation notes (this repo) — VALIDATED on real RDMA

The backend is `src/daemon/export_libfabric.cpp`, compiled only with
`-DOFLD_WITH_LIBFABRIC=ON -DNORTH_COMM_DIR=<north_comm>`; otherwise
`PullTensor(libfabric)` falls back to TCP. It wraps `north_comm`
(`GetNetwork` / `LibfabricEndpointNode` / `Buffer`).

Wire mapping:
- `PullTensor.target_descriptor` for libfabric = `"<recv_nic>|<ib_address_hex>"`
  — the rollout listener's NIC and north_comm `IbAddress`.
- The daemon opens its OWN local NIC via `OFLD_LIBFABRIC_NIC` (falls back to the
  descriptor NIC on a homogeneous fabric) and connects to the receiver's
  advertised `IbAddress`.
- The object is sent as a 1-D `uint8` north_comm **Tensor** buffer (DLPack-able),
  NOT a `Bytes` buffer — `Bytes` round-trips through a UTF-8 `str` and corrupts
  binary weights.

Build linkage gotchas (see CMakeLists `OFLD_WITH_LIBFABRIC` block):
- `libnorth_comm.so` is a pybind extension → the daemon must link the matching
  `libpython` (north_comm here is built for CPython 3.12; override with
  `-DOFLD_PYTHON_LIB=`).
- north_comm public headers need C++20 → only `export_libfabric.cpp` is compiled
  at `-std=c++20` (rest of the daemon stays C++17); dlpack + built libfabric
  headers come from `NORTH_COMM_DIR/3rdparty` and `NORTH_COMM_DIR/build`.
- Runtime `LD_LIBRARY_PATH` needs libfabric's build lib, `/opt/gdrcopy/lib`
  (libfabric was built `--with-gdrcopy`), and the py3.12 lib dir.

Validated end-to-end (`tests/python/{trainer_seal,rollout_libfabric_recv}.py`
driven by `run_{trainer,rollout}_node.sh`): trainer+daemon on one node
canonical-evict + seal+promote a `POLICY_ROLLOUT` version; a rollout client on a
SECOND node discovers the sealed version + manifest over the TCP control port,
then pulls each object over libfabric RDMA — received bytes are byte-exact
against the trainer's source tensors. On this cluster the working RDMA domain is
`mlx5_bond_1` (the RoCE bond); bare `mlx5_N` domains either aren't accepted by
north_comm or don't complete the rxm connection handshake. Confirm a NIC with
north_comm's own `benchmark/cpu_send_recv.py --server-nic X --client-nic X`
before use.
