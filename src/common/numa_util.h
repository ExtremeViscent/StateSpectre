// NUMA + GPU topology helpers, wrapping libnuma and sysfs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace offload {

// True if the kernel/libnuma reports NUMA available.
bool numa_available_ok();

// Highest NUMA node id (>= 0). Returns 0 if NUMA unavailable.
int numa_max_node_id();

// Bind the pages of [addr, addr+len) to the given NUMA node using mbind() with
// MPOL_BIND. Optionally MPOL_MF_MOVE to migrate already-faulted pages.
// Returns 0 on success, -errno on failure. Non-fatal: caller may proceed with
// default policy if this fails (logged by caller).
int bind_range_to_node(void* addr, size_t len, int numa_node, bool move_existing);

// Touch every page in [addr, addr+len) from a thread bound to numa_node, so
// first-touch places the physical pages on that node. len is walked at page
// granularity. Used at arena creation to localize pinned memory.
void first_touch_on_node(void* addr, size_t len, int numa_node);

// Resolve the NUMA node a GPU is attached to, via /sys PCI topology.
// Returns the node id, or -1 if it cannot be determined.
int gpu_numa_node(int cuda_device_index);

// Pin the calling thread to run on CPUs belonging to numa_node. Best-effort.
void pin_thread_to_node(int numa_node);

}  // namespace offload
