# RPC Semantics

## Transport

v1 recommendation:

```text
Unix domain socket + protobuf
SCM_RIGHTS for fd passing during RegisterRank
```

RPC overhead is not expected to matter for 80 GB transfers. If many small tensors dominate, migrate completions to shared-memory rings in v2.

## RegisterRank

Creates a rank session and returns:

```text
rank_epoch
control fd
data arena fds
arena metadata
```

The daemon must invalidate all prior sessions with the same `(rank_id, pid)` or same rank id when a new epoch is issued.

## RequestOffload

Checks:

```text
rank epoch valid
nbytes > 0
GPU/NUMA allowed
alignment allowed
pinned capacity available or drain/backpressure possible
```

Actions:

```text
allocate slot
create lease
populate slot metadata
FREE -> RESERVED_D2H
```

Returns slot offset, not pointer.

## MarkD2HSubmitted

Checks:

```text
lease exists
rank owns lease
slot matches lease
state == RESERVED_D2H
```

Action:

```text
RESERVED_D2H -> D2H_IN_FLIGHT
```

This is useful for timeouts and debugging but may be batched or omitted in an optimized path.

## MarkD2HComplete

Called only after rank-side CUDA event has completed.

Checks:

```text
rank epoch valid
lease exists
owner matches
slot matches
tensor_id/version match
state == D2H_IN_FLIGHT
```

Actions:

```text
D2H_IN_FLIGHT -> PINNED_VALID
if version is latest:
  location[tensor_id] = PINNED(slot_id, version)
enqueue background drain if policy says so
```

The rank may release/invalidate the original GPU tensor after local event completion. Daemon ack is for metadata consistency, not CUDA safety.

## RequestPrefetch

If tensor already in pinned valid slot:

```text
return ready=true, slot info
```

If in pageable/NVMe:

```text
allocate pinned slot
READBACK_IN_FLIGHT
start background read/copy into pinned slot
return ready=false
```

Rank must not H2D until ready.

## MarkH2DSubmitted / MarkH2DComplete

H2D submitted:

```text
PINNED_VALID -> H2D_IN_FLIGHT
```

H2D complete:

```text
H2D_IN_FLIGHT -> GPU_VALID or PINNED_VALID depending keep_pinned/cold-copy policy
```

If `keep_pinned=false` and a valid cold copy exists, the pinned slot may be released.

## BatchComplete

Batch completion is the preferred optimized form after v1.

Each completion must still pass the same checks as individual completion RPCs.

## Error handling

Any invalid transition should:

```text
return ok=false
record error metric
optionally mark slot ERROR if corruption is suspected
```

Rank should treat `ok=false` on completion as fatal unless the response explicitly says the completion was stale but harmless.
