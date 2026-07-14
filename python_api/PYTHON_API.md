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
import state_spectre as ss

with ss.offload_context(
    daemon_addr="unix:///tmp/state_spectre.sock",
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
with ss.offload_context(device="cuda:0") as off:
    x = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)
    h = off.evict(x, name="x")
    h.wait_offloaded()
    x = h.restore(device="cuda:0")
```

### Multiple tensors

```python
with ss.offload_context(device="cuda:0") as off:
    handles = off.evict_many([k_cache, v_cache], names=["k", "v"])
    off.wait(handles)
    k_cache, v_cache = off.restore_many(handles, device="cuda:0")
```

## 12. Tensor ownership and the right offload mode

`evict(..., invalidate="set_empty")` frees VRAM by replacing the storage on the
tensor object you pass (`set_data(empty)`). Whether that actually frees memory
depends on who owns the storage. Match the mode to the tensor:

| Tensor shape | Frees VRAM on evict? | Correct API |
|---|---|---|
| **Standalone, sole owner** — optimizer-state tensors, `param.grad` | Yes | `off.evict(t)` — destructive |
| **`.data` view** — `param.data` | Only after you reassign the holder | `off.evict(param.data)` **and** `param.data = h.restore(...)` |
| **Aliased flat buffer** — Megatron DDP `buffer.param_data`/`grad_data`, distributed-optimizer master-param groups, FSDP `FlatParameter` | No (aliases share the storage; `set_empty` mutates a throwaway impl and desyncs them) | **non-destructive** `off.copy()` + `storage().resize_(0)`, then `storage().resize_(nbytes)` + `h.restore_into(buf)` |

For the aliased-buffer case, preserve the **Storage object** (resize it; never
reassign it) so every alias/view keeps observing the storage's data pointer:

```python
# Offload half — free VRAM without breaking aliases.
h = off.copy(buf, wait=True)          # D2H; buf untouched. wait=True is REQUIRED
buf.untyped_storage().resize_(0)      # free VRAM, keep the Storage object

# ... training proceeds without buf resident ...

# Restore half — realloc the SAME storage, H2D back into it.
buf.untyped_storage().resize_(h.nbytes)
h.restore_into(buf)                   # in-place H2D; aliases see the data again
```

`wait=True` on the `copy` is mandatory: the D2H must complete before
`resize_(0)` frees the source. `restore_into` allocates nothing and requires
`dst` to already be sized to the offloaded byte count.

## 13. Canonical offload/reload round-trip (trainer reload with DP dedup)

The canonical layer dedups the *write* across data-parallel replicas
(`ATTACHED_EXISTING` → no D2H), and `CanonicalHandle.restore_into` provides the
matching *read-back* so a symmetric offload→reload (e.g. free params during
rollout, then reload them onto the same trainer GPUs) shares one host copy —
every replica, including one that only attached, reads the same bytes back by
`object_id`:

```python
key = off.canonical_key(model_role="policy_trainable", model_version=step,
                        param_name=name, tensor=w, pp_rank=pp, tp_rank=tp)
h = off.canonical_evict(w, key, destructive=True)   # create OR attach (dedup)
# ... rollout runs with VRAM freed ...
h.restore_into(out)                                  # reload the shared copy by object_id
```

`restore_into` is read-only w.r.t. the object (the daemon holds it resident for
the H2D and never frees/migrates it), so all replicas can reload concurrently
and repeatedly. It resolves the object from whichever tier it currently lives in
(pinned / pageable / NVMe), warming a cold object back to pinned on first read.
For aliased flat buffers, combine with the storage-resize recipe in §12:
`off.copy`/`canonical_evict` → `storage().resize_(0)` → `resize_(nbytes)` →
`h.restore_into(buf)`.

## 14. Version lifecycle & GC (bounding host/NVMe over a long run)

Because param bytes change every optimizer step, the offload/reload round-trip
must use a **fresh `model_version` each cycle** (reusing a version would
dedup-attach to stale bytes). The daemon does **not** auto-reclaim old versions
from the refcount alone — `force_drain` only migrates tiers, it does not delete
objects — so a long run must drop old versions explicitly, or host/NVMe grows
unbounded:

```python
h = off.canonical_evict(w, off.canonical_key(..., model_version=step), destructive=True)
# ... rollout, then reload ...
h.restore_into(out)
off.drop_canonical_version("policy_trainable", step - 1)   # reclaim last cycle
```

`off.drop_canonical_version(model_role, model_version)` releases every object of
that version (and its manifest), returning
`{"dropped", "skipped_inflight", "bytes_freed", "message"}`. It honors in-flight
export/restore holds (those objects are skipped and reported — retry after the
transfer finishes) and ignores the advisory attachment refcount (it is a coarse
"this whole version is dead" GC the trainer coordinates).

For the **rollout-publish** path this is automatic: sealing+promoting a new
version prunes sealed versions beyond `manifest.retain_sealed_versions` (oldest
first, never the promoted latest, skipping any mid-transfer). Set it to 0 to
disable auto-retention.
