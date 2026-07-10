# Race Conditions and Correctness Invariants

## 1. GPU storage freed before D2H consumes it

Bad:

```text
launch D2H
immediately invalidate/free tensor storage
```

Correct:

```text
launch D2H
record CUDA event
only invalidate/free original storage after event completion
```

Owner: rank-side agent.

## 2. Daemon drains before D2H is complete

Bad:

```text
rank launches GPU -> pinned
daemon starts pinned -> NVMe before D2H completes
```

Correct:

```text
daemon can drain only after MarkD2HComplete
```

Owner: daemon state machine.

## 3. Slot reused before cold writeback finishes

Bad:

```text
slot has tensor A
daemon writes A -> NVMe
rank receives same slot for tensor B
```

Correct:

```text
DRAIN_IN_FLIGHT slots are not allocatable
```

Owner: daemon allocator.

## 4. Multiple processes write same pinned range

Bad:

```text
process 0 and process 1 both know arena offset and copy into it
```

Correct:

```text
all writes require daemon lease
lease includes owner_rank, owner_pid, rank_epoch, slot_id, version
```

Owner: daemon lease table and rank-side checks.

## 5. Stale version overwrites latest location

Bad:

```text
W version 10 offload completes late after W version 11 exists
location[W] becomes version 10
```

Correct:

```text
location update is conditional on version >= current_latest_version
late stale completions are recorded but do not replace latest location
```

Owner: daemon location table.

## 6. Old rank process reports completion after restart

Bad:

```text
rank id 3 dies and restarts
old process sends MarkD2HComplete for old lease
```

Correct:

```text
all RPCs include rank_epoch
old epoch is rejected
```

Owner: daemon session table.

## 7. View/alias incorrectly assumed to free full storage

Bad:

```python
base = torch.empty(1 << 30, device='cuda')
x = base[:1024]
off.evict(x)  # claims to free base storage
```

Correct:

v1 destructive mode requires full storage ownership:

```text
contiguous
storage_offset == 0
logical nbytes == underlying storage nbytes
```

Views may be copied in compact mode, but no VRAM-free guarantee.

Owner: rank-side Python/C++ validation.

## 8. Autograd uses invalidated tensor

Bad:

```text
tensor participates in autograd graph
destructive evict invalidates storage
backward later expects original tensor
```

Correct:

Default destructive mode rejects tensors requiring autograd safety unless explicitly unsafe or integrated via saved_tensors_hooks.

Owner: Python API.

## 9. Pinned slot latest copy lost

Bad:

```text
D2H complete -> PINNED_VALID
slot marked FREE before pageable/NVMe copy or restore
```

Correct invariant:

```text
If latest valid copy exists only in pinned slot, the slot cannot be freed or overwritten.
```

Owner: daemon slot table.

## 10. H2D restore from slot while readback is in-flight

Bad:

```text
NVMe -> pinned readback still running
rank starts pinned -> GPU H2D
```

Correct:

```text
rank can H2D only from PINNED_VALID slot
READBACK_IN_FLIGHT is not readable
```

Owner: daemon readiness notification and rank validation.
