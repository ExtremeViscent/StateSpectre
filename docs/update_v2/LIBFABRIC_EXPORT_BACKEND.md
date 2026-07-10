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
