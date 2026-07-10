#pragma once

#include <stdint.h>

/*
 * Atomic fields are shared across processes and across the C daemon-side ABI
 * consumers and C++ agent/daemon code. C11 _Atomic and C++ std::atomic have
 * identical size/alignment/representation for the lock-free integer types used
 * here, so we select the right spelling per language to keep one on-disk /
 * in-shm layout. Access is always through explicit atomic load/store with a
 * defined memory order on both sides.
 */
#ifdef __cplusplus
#include <atomic>
#define OFLD_ATOMIC(T) std::atomic<T>
#else
#include <stdatomic.h>
#define OFLD_ATOMIC(T) _Atomic T
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OFLD_MAGIC 0x4F464C44u /* 'OFLD' */
#define OFLD_ABI_VERSION 1u
#define OFLD_CACHELINE 64u

typedef uint64_t ofld_tensor_id_t;
typedef uint64_t ofld_version_t;
typedef uint64_t ofld_lease_id_t;
typedef uint32_t ofld_slot_id_t;
typedef uint64_t ofld_arena_id_t;

typedef enum ofld_slot_state_e {
    OFLD_SLOT_FREE                = 0,
    OFLD_SLOT_RESERVED_D2H        = 1,
    OFLD_SLOT_D2H_IN_FLIGHT       = 2,
    OFLD_SLOT_PINNED_VALID        = 3,
    OFLD_SLOT_DRAIN_IN_FLIGHT     = 4,
    OFLD_SLOT_COLD_VALID          = 5,

    OFLD_SLOT_RESERVED_H2D        = 6,
    OFLD_SLOT_READBACK_IN_FLIGHT  = 7,
    OFLD_SLOT_H2D_IN_FLIGHT       = 8,
    OFLD_SLOT_GPU_VALID           = 9,

    OFLD_SLOT_ERROR               = 255
} ofld_slot_state_t;

typedef enum ofld_location_kind_e {
    OFLD_LOC_NONE      = 0,
    OFLD_LOC_GPU       = 1,
    OFLD_LOC_PINNED    = 2,
    OFLD_LOC_PAGEABLE  = 3,
    OFLD_LOC_NVME      = 4,
    OFLD_LOC_DELETED   = 5,
    OFLD_LOC_ERROR     = 255
} ofld_location_kind_t;

typedef enum ofld_arena_kind_e {
    OFLD_ARENA_DATA_PINNED = 1,
    OFLD_ARENA_CONTROL     = 2,
    OFLD_ARENA_OVERFLOW    = 3
} ofld_arena_kind_t;

typedef enum ofld_flags_e {
    OFLD_FLAG_NONE              = 0,
    OFLD_FLAG_ALLOW_OVERFLOW    = 1ull << 0,
    OFLD_FLAG_ALLOW_REMOTE_NUMA = 1ull << 1,
    OFLD_FLAG_BORROWABLE        = 1ull << 2,
    OFLD_FLAG_DESTRUCTIVE       = 1ull << 3,
    OFLD_FLAG_VIEW_COMPACT      = 1ull << 4,
    OFLD_FLAG_UNSAFE_AUTOGRAD   = 1ull << 5
} ofld_flags_t;

/*
 * ABI rule: never store process-local pointers in shared memory.
 * Store arena_id + arena_offset. Each process computes its own local pointer.
 */
typedef struct ofld_arena_desc_s {
    ofld_arena_id_t arena_id;
    uint64_t base_offset;
    uint64_t size;
    uint32_t numa_node;
    uint32_t kind; /* ofld_arena_kind_t */
    uint32_t registration_granularity;
    uint32_t allocation_granularity;
    uint32_t preferred_gpu;
    uint64_t flags;
    uint8_t  reserved[24];
} ofld_arena_desc_t;

/* Keep each slot entry isolated enough to avoid severe false sharing. */
typedef struct ofld_slot_entry_s {
    OFLD_ATOMIC(uint32_t) state; /* ofld_slot_state_t */
    ofld_slot_id_t slot_id;
    uint32_t numa_node;
    uint32_t gpu_preferred;

    ofld_arena_id_t arena_id;
    uint64_t arena_offset;
    uint64_t capacity;
    uint64_t nbytes;

    ofld_tensor_id_t tensor_id;
    ofld_version_t version;
    ofld_lease_id_t lease_id;
    uint64_t owner_rank;

    uint64_t owner_pid;
    uint64_t rank_epoch;
    uint64_t flags;
    uint64_t cold_ref;

    uint64_t checksum;
    uint64_t last_touch_ns;
    uint64_t submit_seq;
    uint64_t complete_seq;

    uint8_t reserved[64];
} __attribute__((aligned(OFLD_CACHELINE))) ofld_slot_entry_t;

typedef struct ofld_slot_table_header_s {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t num_slots;
    uint32_t num_arenas;

    uint64_t generation;
    uint64_t daemon_pid;
    uint64_t daemon_epoch;
    OFLD_ATOMIC(uint64_t) heartbeat_ns;

    uint64_t flags;
    uint64_t arena_desc_offset;
    uint64_t slot_entries_offset;
    uint64_t control_ring_offset;

    uint8_t reserved[64];
} __attribute__((aligned(OFLD_CACHELINE))) ofld_slot_table_header_t;

typedef struct ofld_completion_entry_s {
    ofld_lease_id_t lease_id;
    ofld_tensor_id_t tensor_id;
    ofld_version_t version;
    ofld_slot_id_t slot_id;
    uint32_t event_type;
    uint64_t timestamp_ns;
} ofld_completion_entry_t;

#ifdef __cplusplus
}
#endif
