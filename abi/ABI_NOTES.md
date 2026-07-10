# Shared-Memory ABI Notes

## 1. Offsets, not pointers

Do not put raw pointers into the shared ABI.

Bad:

```cpp
void* ptr;
```

Good:

```cpp
uint64_t arena_id;
uint64_t arena_offset;
```

Each process computes:

```cpp
void* ptr = mapped_arena_base[arena_id] + arena_offset;
```

The daemon and rank processes have different virtual addresses for the same fd mapping.

## 2. Control vs data regions

The control table is normal shared memory and does not need CUDA registration.

Data arenas are fd-backed shared memory mapped by rank processes and registered with `cudaHostRegister()` on rank-local virtual addresses.

## 3. State authority

The daemon is the authority for slot ownership and allocator decisions.

Ranks can read slot metadata, validate leases, and write completion information only through agreed RPC or ring protocols.

## 4. Atomic state

Slot state is atomic for observability and safe transitions, but v1 should still route state transitions through daemon RPC handlers. Avoid arbitrary rank-side CAS transitions except for debug/assertion use.

## 5. Rank epochs

Every session gets a `rank_epoch`. All leases and completions include it. This prevents stale completions from old rank processes after restarts.

## 6. Alignment

Recommended:

```text
allocation granularity: 2 MB or larger for large tensors
registration granularity: 256 MB / 512 MB / 1 GB
slot alignment: at least 256 B, preferably page-aligned
```

Use larger granularity for NVMe-friendly sequential writeback.
