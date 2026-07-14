// Layout of the control shared-memory region.
//
// The control mapping is a single fd-backed region shared between daemon and
// all rank processes. Layout (all offsets from the mapping base):
//
//   [ ofld_slot_table_header_t              ]  at 0
//   [ ofld_arena_desc_t   * num_arenas      ]  at header.arena_desc_offset
//   [ ofld_slot_entry_t   * num_slots       ]  at header.slot_entries_offset
//   [ control ring (reserved, unused in v1) ]  at header.control_ring_offset
//
// The daemon owns all writes to slot state (routed through RPC handlers).
// Ranks read slot metadata (arena_id/offset/state) to compute local pointers
// and to validate readiness. Per docs/design/abi-notes.md, ranks do not perform arbitrary
// CAS transitions in v1.

#pragma once

#include <cstddef>
#include <cstdint>

#include "offload_abi.h"

namespace offload {

// Round up to a cacheline for clean separation of regions.
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

struct ControlLayout {
    uint64_t header_offset = 0;
    uint64_t arena_desc_offset = 0;
    uint64_t slot_entries_offset = 0;
    uint64_t control_ring_offset = 0;
    uint64_t control_ring_bytes = 0;   // total bytes for ALL per-rank rings
    uint64_t total_bytes = 0;
    uint32_t num_arenas = 0;
    uint32_t num_slots = 0;
    uint32_t max_ranks = 0;            // number of per-rank completion rings
    uint32_t ring_stride = 0;          // bytes per rank ring (header+entries)
};

// Compute the layout for a given arena/slot count. ring_bytes may be 0.
inline ControlLayout compute_control_layout(uint32_t num_arenas,
                                            uint32_t num_slots,
                                            uint64_t ring_bytes = 0) {
    ControlLayout L;
    L.num_arenas = num_arenas;
    L.num_slots = num_slots;
    L.control_ring_bytes = ring_bytes;

    uint64_t off = 0;
    L.header_offset = off;
    off += sizeof(ofld_slot_table_header_t);
    off = align_up(off, OFLD_CACHELINE);

    L.arena_desc_offset = off;
    off += static_cast<uint64_t>(num_arenas) * sizeof(ofld_arena_desc_t);
    off = align_up(off, OFLD_CACHELINE);

    L.slot_entries_offset = off;
    off += static_cast<uint64_t>(num_slots) * sizeof(ofld_slot_entry_t);
    off = align_up(off, OFLD_CACHELINE);

    L.control_ring_offset = off;
    off += ring_bytes;
    off = align_up(off, 4096);  // page-align total

    L.total_bytes = off;
    return L;
}

// Typed accessors into a mapped control base pointer.
inline ofld_slot_table_header_t* control_header(void* base) {
    return reinterpret_cast<ofld_slot_table_header_t*>(base);
}
inline ofld_arena_desc_t* control_arena_descs(void* base,
                                              const ofld_slot_table_header_t* h) {
    return reinterpret_cast<ofld_arena_desc_t*>(
        static_cast<char*>(base) + h->arena_desc_offset);
}
inline ofld_slot_entry_t* control_slots(void* base,
                                        const ofld_slot_table_header_t* h) {
    return reinterpret_cast<ofld_slot_entry_t*>(
        static_cast<char*>(base) + h->slot_entries_offset);
}

}  // namespace offload
