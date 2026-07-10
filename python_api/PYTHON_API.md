# Python User Interface

## 1. Design Goal

The API must work with normal PyTorch code without requiring tensors to be created by this runtime.

Therefore, the primary user model is on-demand adoption and destructive eviction:

```python
handle = off.evict(existing_cuda_tensor)
```

The user/runtime guarantees that the original tensor is not accessed after destructive eviction until restored through the handle.

## 2. Context manager role

The context manager manages the runtime session, not tensor creation.

```python
import fastoffload as fo

with fo.offload_context(
    daemon_addr="unix:///tmp/fastoffload.sock",
    device="cuda:0",
    rank=local_rank,
    register_policy="lazy_chunked",
    registration_chunk_mb=512,
    target_tier="pageable_then_nvme",
    invalidate_mode="set_empty",
) as off:
    x = make_tensor_with_normal_pytorch()
    h = off.evict(x, name="x", stream=torch.cuda.current_stream())
    x2 = h.restore(device="cuda:0", stream=torch.cuda.current_stream())
```

The context performs:

```text
connect daemon
receive fds
mmap control/data regions
cudaHostRegister chunks lazily/eagerly
start completion progress thread
cleanup leases on exit
```

## 3. Copy vs evict

Two modes must be explicit.

```python
h = off.copy(tensor)   # non-destructive; original remains valid
h = off.evict(tensor)  # destructive; original is invalidated after D2H completes
```

`evict()` is the performance path for freeing VRAM.

## 4. On-demand management

Optional tracking without immediate offload:

```python
record = off.manage(tensor, name="tmp")
# ... later
handle = record.evict(stream=s)
```

Convenience one-shot:

```python
handle = off.evict(tensor, name="tmp")
```

Semantically:

```python
record = off.manage(tensor)
handle = record.evict()
```

## 5. OffloadHandle

```python
class OffloadHandle:
    tensor_id: int
    version: int
    name: str | None
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device_origin: torch.device
    nbytes: int

    def ready(self) -> bool: ...
    def wait_offloaded(self) -> None: ...
    def location(self) -> str: ...
    def prefetch(self, device=None, stream=None) -> "PrefetchHandle": ...
    def restore(self, device=None, stream=None, wait=True) -> torch.Tensor: ...
    def discard(self) -> None: ...
```

`restore()` returns a new tensor by default. In-place restore is optional/unsafe and not v1 default.

## 6. Destructive eviction semantics

```python
h = off.evict(tensor, stream=s, invalidate="set_empty", wait=False)
```

Steps:

```text
1. validate tensor
2. capture metadata
3. request daemon lease
4. launch D2H into pinned slot
5. record CUDA event
6. rank progress thread waits/queries event
7. after event completion:
   - notify daemon
   - replace original tensor storage with empty storage
   - optionally trigger allocator/cache release policy
8. return handle immediately unless wait=True
```

The user must not use the original tensor after `evict()` is called, unless they explicitly wait and know invalidation is complete. The only supported way to use the data again is `handle.restore()`.

## 7. Validation defaults

For v1 destructive eviction, require:

```text
tensor.is_cuda
tensor.is_contiguous()
tensor.storage_offset() == 0
tensor logical nbytes == underlying storage nbytes
not tensor.requires_grad, unless unsafe_autograd=True
```

Views can be supported explicitly:

```python
h = off.evict(tensor, allow_views=True, compact=True)
```

View compact mode copies logical tensor bytes and restores as a compact contiguous tensor. It does not guarantee freeing base storage.

## 8. Batch API

Many small tensors need batching.

```python
handles = off.evict_many(
    tensors,
    names=None,
    stream=torch.cuda.current_stream(),
    invalidate="set_empty",
    require_own_storage=True,
)

off.wait(handles)
restored = off.restore_many(handles, device="cuda:0", stream=s)
```

Batching should coalesce lease RPCs and completion reports.

## 9. Unsafe modes

```python
with off.unsafe_destructive_mode():
    h = off.evict(tensor, unsafe_autograd=True)
```

Unsafe mode acknowledges:

```text
- autograd correctness is caller's responsibility
- aliases/views may exist
- caller guarantees no direct tensor use after eviction
```

## 10. Debugging

```python
print(off.summary())
off.assert_no_inflight()
off.slot_table(limit=20)

with off.trace("/tmp/offload_trace.json"):
    ...
```

Summary should include:

```text
rank/gpu/numa
registered pinned bytes
used pinned bytes
in-flight D2H/H2D bytes
draining bytes
pageable/NVMe usage
D2H/H2D bandwidth percentiles
blocked reservation count
remote-NUMA allocation count
```

## 11. Minimal v1 examples

### Single tensor

```python
with fo.offload_context(device="cuda:0") as off:
    x = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)
    h = off.evict(x, name="x")
    h.wait_offloaded()
    x = h.restore(device="cuda:0")
```

### Multiple tensors

```python
with fo.offload_context(device="cuda:0") as off:
    handles = off.evict_many([k_cache, v_cache], names=["k", "v"])
    off.wait(handles)
    k_cache, v_cache = off.restore_many(handles, device="cuda:0")
```
