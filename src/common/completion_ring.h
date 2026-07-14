// Lock-free single-producer/single-consumer completion ring in shared memory.
//
// Each rank session owns one ring in the control region's ring area. The rank
// (producer) pushes ofld_completion_entry_t records as CUDA events complete;
// the daemon (single consumer for that ring) pops them and applies the slot
// state transitions — with NO socket RPC on the completion hot path. This is
// the v2 optimization from docs/design/rpc-semantics.md ("migrate completions to
// shared-memory rings").
//
// Memory model: classic SPSC ring with head (consumer) and tail (producer)
// indices. The producer publishes an entry by writing the slot then advancing
// tail with release; the consumer reads tail with acquire, reads the entry,
// then advances head with release. Only one producer and one consumer touch a
// given ring, so no CAS is needed. Indices are free-running uint64 counters;
// slot = index % capacity. Capacity MUST be a power of two.
//
// The ring lives entirely in the shared fd mapping, so head/tail are addressed
// relative to the mapping base (offsets, never pointers) per docs/design/abi-notes.md.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "offload_abi.h"

namespace offload {

// Per-ring header, followed immediately by `capacity` ofld_completion_entry_t.
// One RingHeader+entries block per rank. Cacheline-pad head/tail to avoid
// false sharing between producer and consumer.
struct alignas(OFLD_CACHELINE) CompletionRingHeader {
    uint32_t capacity;        // number of entries (power of two)
    uint32_t owner_rank;      // rank_id that owns this ring (producer)
    uint64_t rank_epoch;      // epoch of the owning session; 0 => unassigned

    // Consumer index (daemon). On its own cacheline (padding via alignas).
    alignas(OFLD_CACHELINE) OFLD_ATOMIC(uint64_t) head;
    // Producer index (rank). On its own cacheline.
    alignas(OFLD_CACHELINE) OFLD_ATOMIC(uint64_t) tail;
};

// Total bytes for one ring (header + entries), cacheline-rounded.
inline uint64_t completion_ring_bytes(uint32_t capacity) {
    uint64_t hdr = sizeof(CompletionRingHeader);
    uint64_t ents = static_cast<uint64_t>(capacity) * sizeof(ofld_completion_entry_t);
    uint64_t total = hdr + ents;
    return (total + OFLD_CACHELINE - 1) / OFLD_CACHELINE * OFLD_CACHELINE;
}

// Entries array immediately follows the header.
inline ofld_completion_entry_t* ring_entries(CompletionRingHeader* h) {
    return reinterpret_cast<ofld_completion_entry_t*>(
        reinterpret_cast<char*>(h) + sizeof(CompletionRingHeader));
}

// Producer: try to push one entry. Returns false if the ring is full (caller
// should fall back to an RPC). Single-producer only.
inline bool ring_push(CompletionRingHeader* h, const ofld_completion_entry_t& e) {
    uint64_t tail = h->tail.load(std::memory_order_relaxed);
    uint64_t head = h->head.load(std::memory_order_acquire);
    if (tail - head >= h->capacity) return false;  // full
    ofld_completion_entry_t* ents = ring_entries(h);
    ents[tail & (h->capacity - 1)] = e;
    h->tail.store(tail + 1, std::memory_order_release);
    return true;
}

// Consumer: try to pop one entry. Returns false if empty. Single-consumer only.
inline bool ring_pop(CompletionRingHeader* h, ofld_completion_entry_t* out) {
    uint64_t head = h->head.load(std::memory_order_relaxed);
    uint64_t tail = h->tail.load(std::memory_order_acquire);
    if (head == tail) return false;  // empty
    ofld_completion_entry_t* ents = ring_entries(h);
    *out = ents[head & (h->capacity - 1)];
    h->head.store(head + 1, std::memory_order_release);
    return true;
}

// Number of pending entries (approximate; for metrics/debug).
inline uint64_t ring_pending(const CompletionRingHeader* h) {
    uint64_t tail = h->tail.load(std::memory_order_acquire);
    uint64_t head = h->head.load(std::memory_order_acquire);
    return tail - head;
}

// --- ring-area metadata carried in the slot-table header's reserved[64] ---
//
// We store the ring layout (max_ranks, per-rank stride, ring capacity) in the
// header's reserved bytes so both daemon and rank agree without changing the
// frozen ABI struct layout. Overlay: reserved[0..3]=max_ranks (u32),
// [4..7]=ring_stride (u32), [8..11]=ring_capacity (u32). Written by the daemon
// at startup, read by ranks after mapping the control region.
struct RingAreaMeta {
    uint32_t max_ranks;
    uint32_t ring_stride;   // bytes per rank ring (== completion_ring_bytes(cap))
    uint32_t ring_capacity; // entries per ring
    uint32_t _pad;
};
static_assert(sizeof(RingAreaMeta) <= 64, "RingAreaMeta must fit in reserved[64]");

inline RingAreaMeta* ring_area_meta(ofld_slot_table_header_t* h) {
    return reinterpret_cast<RingAreaMeta*>(h->reserved);
}
inline const RingAreaMeta* ring_area_meta(const ofld_slot_table_header_t* h) {
    return reinterpret_cast<const RingAreaMeta*>(h->reserved);
}

// Pointer to rank `slot`'s ring header within the mapped control region.
// slot is the rank's assigned ring index in [0, max_ranks). Returns nullptr if
// rings are not configured (max_ranks == 0) or the index is out of range.
inline CompletionRingHeader* ring_for_index(void* control_base,
                                            const ofld_slot_table_header_t* h,
                                            uint32_t index) {
    const RingAreaMeta* m = ring_area_meta(h);
    if (m->max_ranks == 0 || index >= m->max_ranks) return nullptr;
    char* base = static_cast<char*>(control_base) + h->control_ring_offset;
    return reinterpret_cast<CompletionRingHeader*>(
        base + static_cast<uint64_t>(index) * m->ring_stride);
}

}  // namespace offload
