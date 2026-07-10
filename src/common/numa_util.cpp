#include "numa_util.h"

#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#include <errno.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "util.h"

namespace offload {

bool numa_available_ok() { return numa_available() >= 0; }

int numa_max_node_id() {
    if (numa_available() < 0) return 0;
    int m = numa_max_node();
    return m < 0 ? 0 : m;
}

int bind_range_to_node(void* addr, size_t len, int numa_node, bool move_existing) {
    if (numa_node < 0) return 0;
    if (numa_available() < 0) return 0;  // no NUMA: nothing to do, treat as ok

    // Build a node mask with just numa_node set. MPOL_BIND enforces allocation
    // on that node. MPOL_MF_MOVE migrates pages already faulted in.
    unsigned long maxnode = static_cast<unsigned long>(numa_max_node()) + 1;
    // node mask needs ceil(maxnode/64) longs; round to at least 1.
    size_t nlongs = (maxnode + 8 * sizeof(unsigned long) - 1) /
                    (8 * sizeof(unsigned long));
    if (nlongs == 0) nlongs = 1;
    std::vector<unsigned long> mask(nlongs, 0ul);
    mask[numa_node / (8 * sizeof(unsigned long))] |=
        (1ul << (numa_node % (8 * sizeof(unsigned long))));

    unsigned flags = move_existing ? (MPOL_MF_MOVE | MPOL_MF_STRICT) : 0;
    // mbind's maxnode counts bits, and expects +1 for the mask length convention.
    long rc = mbind(addr, len, MPOL_BIND, mask.data(),
                    maxnode + 1, flags);
    if (rc != 0) {
        int e = errno;
        OFLD_WARN("numa", "mbind(node=%d,len=%zu) failed: %s",
                  numa_node, len, strerror(e));
        return -e;
    }
    return 0;
}

void first_touch_on_node(void* addr, size_t len, int numa_node) {
    // Run the touch from a thread already pinned to the node (caller decides),
    // combined with MPOL_BIND above, so pages land locally. We write one byte
    // per page. Using volatile to prevent the loop being optimized away.
    long pgsz = sysconf(_SC_PAGESIZE);
    if (pgsz <= 0) pgsz = 4096;
    volatile char* p = static_cast<volatile char*>(addr);
    for (size_t off = 0; off < len; off += static_cast<size_t>(pgsz)) {
        p[off] = 0;
    }
    (void)numa_node;
}

int gpu_numa_node(int cuda_device_index) {
    // Resolve the PCI bus id of the CUDA device, then read
    // /sys/bus/pci/devices/<bdf>/numa_node. We avoid a hard CUDA dependency
    // here by reading the ordering the driver exposes through
    // /proc/driver/nvidia if present; otherwise fall back to sysfs scan.
    //
    // Primary path: CUDA runtime gives PCI bus/device via cudaDeviceGetPCIBusId,
    // but numa_util must not link CUDA. So we consult sysfs for NVIDIA devices
    // ordered by PCI address, which matches CUDA_DEVICE_ORDER=PCI_BUS_ID.
    //
    // Build a sorted list of NVIDIA GPU BDFs and index into it.
    const char* devdir = "/sys/bus/pci/devices";
    // NVIDIA vendor id is 0x10de; class 0x030000/0x030200 (VGA/3D controller).
    std::vector<std::string> gpu_bdfs;
    // Read directory manually.
    if (FILE* pf = popen(
            "ls -1 /sys/bus/pci/devices 2>/dev/null", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), pf)) {
            std::string bdf(line);
            while (!bdf.empty() && (bdf.back() == '\n' || bdf.back() == ' '))
                bdf.pop_back();
            if (bdf.empty()) continue;
            std::string vpath = std::string(devdir) + "/" + bdf + "/vendor";
            FILE* vf = fopen(vpath.c_str(), "r");
            if (!vf) continue;
            char vbuf[32] = {0};
            if (fgets(vbuf, sizeof(vbuf), vf)) {
                if (strncmp(vbuf, "0x10de", 6) == 0) {
                    // check class is a display controller (0x03xxxx)
                    std::string cpath = std::string(devdir) + "/" + bdf + "/class";
                    FILE* cf = fopen(cpath.c_str(), "r");
                    if (cf) {
                        char cbuf[32] = {0};
                        if (fgets(cbuf, sizeof(cbuf), cf) &&
                            strncmp(cbuf, "0x03", 4) == 0) {
                            gpu_bdfs.push_back(bdf);
                        }
                        fclose(cf);
                    }
                }
            }
            fclose(vf);
        }
        pclose(pf);
    }
    std::sort(gpu_bdfs.begin(), gpu_bdfs.end());
    if (cuda_device_index < 0 ||
        cuda_device_index >= static_cast<int>(gpu_bdfs.size())) {
        return -1;
    }
    std::string npath =
        std::string(devdir) + "/" + gpu_bdfs[cuda_device_index] + "/numa_node";
    FILE* nf = fopen(npath.c_str(), "r");
    if (!nf) return -1;
    int node = -1;
    if (fscanf(nf, "%d", &node) != 1) node = -1;
    fclose(nf);
    return node;
}

void pin_thread_to_node(int numa_node) {
    if (numa_node < 0 || numa_available() < 0) return;
    struct bitmask* cpus = numa_allocate_cpumask();
    if (!cpus) return;
    if (numa_node_to_cpus(numa_node, cpus) == 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        int ncpu = numa_num_configured_cpus();
        bool any = false;
        for (int c = 0; c < ncpu; ++c) {
            if (numa_bitmask_isbitset(cpus, c)) { CPU_SET(c, &set); any = true; }
        }
        if (any) sched_setaffinity(0, sizeof(set), &set);
    }
    numa_free_cpumask(cpus);
}

}  // namespace offload
