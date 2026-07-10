// Unit test: control-region layout math + slot/pointer translation (TEST_PLAN §1).
#include "control_layout.h"
#include "offload_abi.h"
#include "test_harness.h"

#include <cstdlib>
#include <vector>

using namespace offload;
using namespace ofldtest;

static void test_layout_monotonic_and_aligned() {
    ControlLayout L = compute_control_layout(/*arenas=*/4, /*slots=*/1000);
    // Regions must be in ascending, non-overlapping order.
    CHECK(L.header_offset == 0);
    CHECK(L.arena_desc_offset >= sizeof(ofld_slot_table_header_t));
    CHECK(L.slot_entries_offset >=
          L.arena_desc_offset + 4 * sizeof(ofld_arena_desc_t));
    CHECK(L.total_bytes >=
          L.slot_entries_offset + 1000 * sizeof(ofld_slot_entry_t));
    // Cacheline alignment of the region starts.
    CHECK_EQ(L.arena_desc_offset % OFLD_CACHELINE, 0ull);
    CHECK_EQ(L.slot_entries_offset % OFLD_CACHELINE, 0ull);
    // Total is page aligned.
    CHECK_EQ(L.total_bytes % 4096, 0ull);
    CHECK_EQ(L.num_arenas, 4u);
    CHECK_EQ(L.num_slots, 1000u);
}

static void test_accessors_translate() {
    // Build a real backing buffer and lay out header + arenas + slots into it,
    // then read them back through the typed accessors.
    ControlLayout L = compute_control_layout(2, 8);
    std::vector<uint8_t> buf(L.total_bytes, 0);
    void* base = buf.data();

    auto* h = control_header(base);
    h->magic = OFLD_MAGIC;
    h->abi_version = OFLD_ABI_VERSION;
    h->num_arenas = L.num_arenas;
    h->num_slots = L.num_slots;
    h->arena_desc_offset = L.arena_desc_offset;
    h->slot_entries_offset = L.slot_entries_offset;
    h->control_ring_offset = L.control_ring_offset;

    auto* arenas = control_arena_descs(base, h);
    arenas[0].arena_id = 100; arenas[0].size = (64ull << 20);
    arenas[1].arena_id = 200; arenas[1].size = (32ull << 20);

    auto* slots = control_slots(base, h);
    for (uint32_t i = 0; i < 8; ++i) {
        slots[i].slot_id = i;
        slots[i].arena_id = (i < 4) ? 100 : 200;
        slots[i].arena_offset = (uint64_t)i * (2ull << 20);
        slots[i].capacity = (2ull << 20);
        slots[i].state.store(OFLD_SLOT_FREE, std::memory_order_relaxed);
    }

    // Read back through accessors.
    auto* h2 = control_header(base);
    CHECK_EQ(h2->magic, OFLD_MAGIC);
    CHECK_EQ(h2->num_slots, 8u);
    auto* a2 = control_arena_descs(base, h2);
    CHECK_EQ(a2[0].arena_id, 100ull);
    CHECK_EQ(a2[1].arena_id, 200ull);
    auto* s2 = control_slots(base, h2);
    CHECK_EQ(s2[5].slot_id, 5u);
    CHECK_EQ(s2[5].arena_id, 200ull);
    CHECK_EQ(s2[5].arena_offset, (uint64_t)5 * (2ull << 20));
    CHECK_EQ(s2[7].state.load(std::memory_order_relaxed),
             (uint32_t)OFLD_SLOT_FREE);

    // Slot pointer translation (as a rank would compute it): the offset of a
    // slot inside its arena is arena_offset; the arena base is process-local.
    // Verify offsets are distinct and within arena bounds.
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(s2[i].arena_offset + s2[i].capacity <= a2[0].size);
    }
}

static void test_slot_entry_cacheline_aligned() {
    // The ABI aligns slot entries to a cacheline to avoid false sharing.
    CHECK_EQ(sizeof(ofld_slot_entry_t) % OFLD_CACHELINE, 0ull);
    CHECK_EQ(alignof(ofld_slot_entry_t), (size_t)OFLD_CACHELINE);
}

int main() {
    RUN(test_layout_monotonic_and_aligned);
    RUN(test_accessors_translate);
    RUN(test_slot_entry_cacheline_aligned);
    return summary("layout");
}
