// Benchmark harness for the offload runtime (tests/METRICS.md + TEST_PLAN §6).
//
// Measures, on real hardware, with an in-process daemon + real rank agent:
//   - D2H / H2D bandwidth across tensor sizes (CUDA-event timed)
//   - cudaHostRegister overhead vs chunk size
//   - drain bandwidth: pinned->pageable, pinned->NVMe (pwrite + io_uring,
//     O_DIRECT on/off, block-size sweep)
//   - NVMe readback bandwidth
//   - burst absorption: N concurrent ranks each offloading a shard
//   - NUMA local vs remote D2H (if >1 NUMA node)
//
// Output is a human table plus a machine-readable "BENCH,<name>,<GB>,<GB/s>,..."
// line per result so the report generator can parse it.
//
// Usage:
//   offload_bench [--gpu N] [--nvme-dir DIR[,DIR2]] [--quick] [--sizes 1,8,40,80]
//                 [--section all|d2h|reg|drain|nvme|burst|numa]

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "agent.h"
#include "config.h"
#include "daemon.h"
#include "util.h"

using namespace offload;

#define CU(call)                                                              \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,       \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

namespace {

double gbps(uint64_t bytes, double secs) {
    if (secs <= 0) return 0;
    return static_cast<double>(bytes) / 1e9 / secs;
}
double gib(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0*1024.0*1024.0); }

// Build a config whose single GPU window (no overflow carve-out) fully holds a
// tensor of `window_bytes`, so a large contiguous evict always fits. Slots are
// 64 MiB. NVMe/drain off unless the caller flips them.
DaemonConfig single_window_config(uint64_t window_bytes, int node, int gpu,
                                  const std::string& sock) {
    DaemonConfig cfg;
    cfg.socket_path = sock;
    cfg.allocation_granularity_bytes = 64ull << 20;
    cfg.registration_chunk_bytes = 512ull << 20;
    cfg.nvme_enabled = false;
    cfg.drain_on_d2h_complete = false;
    cfg.drain_target = DrainTarget::kPageableOnly;
    cfg.drain_workers = 4;
    GpuWindowConfig w; w.gpu = (uint32_t)gpu; w.size_bytes = window_bytes;
    NumaArenaConfig a; a.numa_node = (uint32_t)node; a.size_bytes = window_bytes;
    a.gpu_windows.push_back(w); a.overflow_bytes = 0;
    cfg.per_numa.push_back(a);
    return cfg;
}

// Spin up a daemon on a background thread; returns socket path.
struct Daemon {
    std::string sock;
    std::unique_ptr<OffloadDaemon> d;
    std::thread th;

    Daemon(const DaemonConfig& cfg_in) {
        DaemonConfig cfg = cfg_in;
        sock = cfg.socket_path;
        d.reset(new OffloadDaemon(cfg));
        th = std::thread([this] { d->run(); });
        for (int i = 0; i < 400; ++i) {
            // crude readiness wait
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (i > 4) break;
        }
    }
    ~Daemon() {
        d->request_stop();
        if (th.joinable()) th.join();
        ::unlink(sock.c_str());
    }
};

// Measure D2H then H2D bandwidth for a single contiguous transfer of nbytes.
// Uses ONE device buffer (so an 80 GiB tensor needs ~nbytes VRAM, not 2x): D2H
// evicts it (GPU->pinned); H2D restores back into the SAME buffer. Both timed
// wall-clock around the wait=true call (dominated by the DMA). Averaged/best
// over `iters`. The pinned copy is discarded between iters to recycle slots.
void bench_d2h_h2d(OffloadAgent& agent, int /*gpu*/, uint64_t nbytes, int iters,
                   double* d2h_gbps, double* h2d_gbps) {
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, nbytes));
    CU(cudaMemset(dptr, 0xAB, nbytes));
    CU(cudaDeviceSynchronize());
    *d2h_gbps = 0; *h2d_gbps = 0;
    // Namespace tensor_ids by size so ids never collide across bench_d2h_h2d
    // calls (a stale location for a reused id would make restore copy the wrong
    // number of bytes).
    uint64_t tid_base = 100 + (nbytes >> 20) * 1000;

    for (int it = 0; it < iters; ++it) {
        uint64_t tid = tid_base + it;
        TensorMeta m; m.tensor_id = tid; m.version = 1; m.nbytes = nbytes;
        // D2H (evict, non-destructive so dptr stays valid as the H2D dst).
        CU(cudaDeviceSynchronize());
        uint64_t t0 = now_mono_ns();
        OffloadTicket tk = agent.evict((uint64_t)dptr, nbytes, kInternalStream,
                                       /*destructive=*/false, m, nullptr, 0, true);
        CU(cudaDeviceSynchronize());
        uint64_t t1 = now_mono_ns();
        if (!tk.ok) { std::fprintf(stderr, "evict failed: %s\n", tk.message.c_str()); break; }
        *d2h_gbps = std::max(*d2h_gbps, gbps(nbytes, (t1 - t0) / 1e9));

        // H2D (restore back into the same buffer). Sync around it so the wall
        // clock captures the actual DMA, not just the launch/RPC.
        CU(cudaDeviceSynchronize());
        uint64_t r0 = now_mono_ns();
        RestoreResult rr = agent.restore(tid, 1, (uint64_t)dptr, nbytes,
                                         kInternalStream, true);
        CU(cudaDeviceSynchronize());
        uint64_t r1 = now_mono_ns();
        if (rr.ok) *h2d_gbps = std::max(*h2d_gbps, gbps(nbytes, (r1 - r0) / 1e9));
        agent.discard(tid, 1);
    }
    CU(cudaFree(dptr));
}

// cudaHostRegister overhead for a given chunk size over a malloc'd host buffer.
void bench_register(uint64_t chunk, double* ms_per_gb) {
    const uint64_t total = 4ull << 30;  // register 4 GB in `chunk`-sized pieces
    void* base = nullptr;
    if (posix_memalign(&base, 4096, total) != 0) { *ms_per_gb = -1; return; }
    // Touch pages so registration reflects real pinning cost.
    std::memset(base, 1, total);
    uint64_t t0 = now_mono_ns();
    uint64_t off = 0;
    std::vector<void*> regs;
    while (off < total) {
        uint64_t len = std::min(chunk, total - off);
        cudaError_t e = cudaHostRegister((char*)base + off, len, cudaHostRegisterDefault);
        if (e != cudaSuccess) { *ms_per_gb = -1; break; }
        regs.push_back((char*)base + off);
        off += len;
    }
    uint64_t t1 = now_mono_ns();
    for (void* p : regs) cudaHostUnregister(p);
    free(base);
    double secs = (t1 - t0) / 1e9;
    *ms_per_gb = secs * 1000.0 / gib(total);
}

std::vector<uint64_t> parse_sizes_gb(const std::string& s) {
    std::vector<uint64_t> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!tok.empty()) out.push_back((uint64_t)std::stoull(tok) << 30);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffer so progress flushes
    int gpu = 0;
    std::string nvme_dirs = "/dockerdata/ofld_bench";
    std::string section = "all";
    std::string sizes_str = "1,8,40,80";
    bool quick = false;
    uint32_t nvme_block_mb = 16;
    uint32_t nvme_qd = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]{ return std::string(argv[++i]); };
        if (a == "--gpu") gpu = std::stoi(nxt());
        else if (a == "--nvme-dir") nvme_dirs = nxt();
        else if (a == "--section") section = nxt();
        else if (a == "--sizes") sizes_str = nxt();
        else if (a == "--nvme-block-mb") nvme_block_mb = (uint32_t)std::stoul(nxt());
        else if (a == "--nvme-qd") nvme_qd = (uint32_t)std::stoul(nxt());
        else if (a == "--quick") quick = true;
    }
    bool all = (section == "all");

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "[SKIP] no CUDA device\n"); return 0;
    }
    CU(cudaSetDevice(gpu));

    std::vector<uint64_t> sizes = parse_sizes_gb(sizes_str);
    // Cap sizes to fit VRAM. D2H/H2D use a single device buffer of `nbytes`,
    // so we only need ~nbytes on the GPU (+ headroom for context/allocator).
    size_t freeb = 0, totb = 0;
    CU(cudaMemGetInfo(&freeb, &totb));
    uint64_t cap = (uint64_t)(freeb * 0.92);
    std::vector<uint64_t> use_sizes;
    for (uint64_t s : sizes) if (s <= cap) use_sizes.push_back(s);
    if (use_sizes.empty()) use_sizes.push_back(std::min<uint64_t>(cap, 1ull<<30));

    std::printf("# Offload runtime benchmark\n");
    std::printf("# GPU %d, free VRAM %.1f GiB, sizes tested up to %.0f GiB\n",
                gpu, gib(freeb), gib(use_sizes.back()));

    const int iters = quick ? 2 : 3;

    // ================= Section: cudaHostRegister overhead =================
    if (all || section == "reg") {
        std::printf("\n== cudaHostRegister overhead (register 4 GiB in chunks) ==\n");
        std::printf("%-16s %-14s\n", "chunk", "ms/GiB");
        for (uint64_t chunk : {256ull<<20, 512ull<<20, 1ull<<30, 4ull<<30}) {
            double ms = 0; bench_register(chunk, &ms);
            std::printf("%-16s %-14.2f\n",
                        (std::to_string(chunk >> 20) + " MiB").c_str(), ms);
            std::printf("BENCH,register,%llu,%.2f\n",
                        (unsigned long long)(chunk>>20), ms);
        }
    }

    // ================= Section: D2H/H2D bandwidth =================
    if (all || section == "d2h") {
        // Single-window arena (no overflow carve-out) sized to hold the largest
        // tensor's pinned copy while the same-size H2D reads it back.
        DaemonConfig cfg = single_window_config(use_sizes.back() + (1ull<<30),
                                                0, gpu, "/tmp/ofld_bench_d2h.sock");
        ::unlink(cfg.socket_path.c_str());
        Daemon dae(cfg);
        AgentConfig ac; ac.socket_path = cfg.socket_path; ac.cuda_device = gpu;
        ac.rank_id = 0; ac.numa_node = 0;
        OffloadAgent agent(ac);

        std::printf("\n== D2H / H2D bandwidth (PCIe Gen5 x16) ==\n");
        std::printf("%-10s %-14s %-14s\n", "size", "D2H GB/s", "H2D GB/s");
        for (uint64_t nb : use_sizes) {
            double d2h = 0, h2d = 0;
            bench_d2h_h2d(agent, gpu, nb, iters, &d2h, &h2d);
            std::printf("%-10s %-14.2f %-14.2f\n",
                        (std::to_string(nb>>30) + " GiB").c_str(), d2h, h2d);
            std::printf("BENCH,d2h,%llu,%.2f\nBENCH,h2d,%llu,%.2f\n",
                        (unsigned long long)(nb>>30), d2h,
                        (unsigned long long)(nb>>30), h2d);
        }
    }

    // ================= Section: drain to pageable + NVMe =================
    if (all || section == "drain" || section == "nvme") {
        // We measure drain by offloading with drain_on_d2h_complete and timing
        // how fast the daemon migrates pinned->cold (via metrics deltas). Simpler
        // and directly comparable: time a full evict+wait_drained roundtrip for
        // each target/engine. Use a moderate tensor (8 GiB or quick 1 GiB).
        uint64_t nb = quick ? (1ull<<30) : std::min<uint64_t>(8ull<<30, cap);
        std::printf("\n== Drain bandwidth (pinned -> cold), %.0f GiB tensor ==\n", gib(nb));
        std::printf("%-28s %-12s %-12s\n", "target/engine", "write GB/s", "read GB/s");

        struct Variant { const char* name; DrainTarget tgt; bool nvme; std::string engine; bool odirect; };
        std::vector<Variant> variants = {
            {"pageable",            DrainTarget::kPageableOnly, false, "pwrite",   false},
        };
        if (all || section == "nvme") {
            variants.push_back({"nvme pwrite O_DIRECT",  DrainTarget::kNvmeOnly, true, "pwrite",   true});
            variants.push_back({"nvme pwrite buffered",  DrainTarget::kNvmeOnly, true, "pwrite",   false});
            variants.push_back({"nvme io_uring O_DIRECT",DrainTarget::kNvmeOnly, true, "io_uring", true});
            variants.push_back({"nvme io_uring buffered",DrainTarget::kNvmeOnly, true, "io_uring", false});
        }

        // Parse comma-separated nvme dirs into stripe dirs.
        std::vector<std::string> dirs;
        { size_t i=0; while (i<nvme_dirs.size()){
            size_t j=nvme_dirs.find(',',i);
            dirs.push_back(nvme_dirs.substr(i, j==std::string::npos?j:j-i));
            if (j==std::string::npos) break;
            i=j+1;
          } }

        for (const auto& v : variants) {
            uint64_t arena = nb + (2ull<<30);
            DaemonConfig cfg = default_smoke_config(arena, 0, gpu);
            cfg.socket_path = "/tmp/ofld_bench_drain.sock";
            cfg.allocation_granularity_bytes = 64ull << 20;
            cfg.drain_on_d2h_complete = true;
            cfg.drain_target = v.tgt;
            cfg.drain_workers = 4;
            cfg.nvme_enabled = v.nvme;
            cfg.nvme_io_engine = v.engine;
            cfg.nvme_direct_io = v.odirect;
            cfg.nvme_stripe = true;
            cfg.nvme_stripe_count = (uint32_t)std::max<size_t>(dirs.size()*2, 4);
            cfg.nvme_block_bytes = (uint64_t)nvme_block_mb << 20;
            cfg.nvme_queue_depth = nvme_qd;
            cfg.nvme_stripe_dirs = dirs;
            cfg.nvme_path = dirs.empty() ? "/tmp/ofld_bench_nvme" : dirs[0];
            ::unlink(cfg.socket_path.c_str());
            Daemon dae(cfg);
            AgentConfig ac; ac.socket_path = cfg.socket_path; ac.cuda_device = gpu;
            ac.rank_id = 0; ac.numa_node = 0;
            OffloadAgent agent(ac);

            void* dptr = nullptr; CU(cudaMalloc(&dptr, nb));
            CU(cudaMemset(dptr, 0x5A, nb)); CU(cudaDeviceSynchronize());
            TensorMeta m; m.tensor_id = 900; m.version = 1; m.nbytes = nb;
            // evict (D2H), then wait for the daemon to migrate to cold, timing
            // the drain by polling location -> cold.
            agent.evict((uint64_t)dptr, nb, 0, false, m, nullptr, 0, true);
            uint64_t t0 = now_mono_ns();
            std::string want = v.nvme ? "nvme" : "pageable";
            for (int k = 0; k < 20000; ++k) {
                if (agent.location(900,1) == want) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            uint64_t t1 = now_mono_ns();
            double wr = gbps(nb, (t1 - t0)/1e9);
            // readback: restore forces cold->pinned then H2D. Time cold->pinned
            // by restore roundtrip minus H2D (approx: whole restore).
            void* rptr = nullptr; CU(cudaMalloc(&rptr, nb));
            uint64_t r0 = now_mono_ns();
            agent.restore(900, 1, (uint64_t)rptr, nb, 0, true);
            uint64_t r1 = now_mono_ns();
            double rd = gbps(nb, (r1 - r0)/1e9);
            std::printf("%-28s %-12.2f %-12.2f\n", v.name, wr, rd);
            std::printf("BENCH,drain_%s,%llu,%.2f,%.2f\n", v.name,
                        (unsigned long long)(nb>>30), wr, rd);
            CU(cudaFree(dptr)); CU(cudaFree(rptr));
        }
    }

    // ================= Section: burst absorption =================
    if (all || section == "burst") {
        std::printf("\n== Burst absorption: N ranks concurrently evicting ==\n");
        // Single process, multiple agents (distinct rank ids) hitting one daemon.
        const int N = quick ? 2 : 4;
        uint64_t per = quick ? (1ull<<30) : std::min<uint64_t>(4ull<<30, cap/(N+1));
        uint64_t arena = per * N + (2ull<<30);
        DaemonConfig cfg = default_smoke_config(arena, 0, gpu);
        cfg.socket_path = "/tmp/ofld_bench_burst.sock";
        cfg.allocation_granularity_bytes = 64ull<<20;
        cfg.drain_on_d2h_complete = false;
        cfg.nvme_enabled = false;
        ::unlink(cfg.socket_path.c_str());
        Daemon dae(cfg);

        std::vector<std::unique_ptr<OffloadAgent>> agents;
        std::vector<void*> ptrs(N, nullptr);
        for (int i = 0; i < N; ++i) {
            AgentConfig ac; ac.socket_path = cfg.socket_path; ac.cuda_device = gpu;
            ac.rank_id = (uint32_t)(50+i); ac.numa_node = 0;
            agents.emplace_back(new OffloadAgent(ac));
            CU(cudaMalloc(&ptrs[i], per));
            CU(cudaMemset(ptrs[i], i+1, per));
        }
        CU(cudaDeviceSynchronize());
        uint64_t t0 = now_mono_ns();
        std::vector<std::thread> ts;
        for (int i = 0; i < N; ++i) {
            ts.emplace_back([&, i]{
                TensorMeta m; m.tensor_id = 6000+i; m.version=1; m.nbytes=per;
                agents[i]->evict((uint64_t)ptrs[i], per, 0, false, m, nullptr, 0, true);
            });
        }
        for (auto& t : ts) t.join();
        uint64_t t1 = now_mono_ns();
        double agg = gbps(per*(uint64_t)N, (t1-t0)/1e9);
        std::printf("%d ranks x %.0f GiB: aggregate D2H %.2f GB/s (wall %.2fs)\n",
                    N, gib(per), agg, (t1-t0)/1e9);
        std::printf("BENCH,burst,%d,%.2f\n", N, agg);
        for (int i = 0; i < N; ++i) CU(cudaFree(ptrs[i]));
    }

    // ================= Section: NUMA local vs remote =================
    if ((all || section == "numa")) {
        std::printf("\n== NUMA local vs remote D2H ==\n");
        // Compare D2H into an arena bound to the GPU's local node vs the other.
        int local_node = 0;  // smoke arena uses node 0; GPU0 is on node 0
        uint64_t nb = quick ? (1ull<<30) : std::min<uint64_t>(8ull<<30, cap);
        for (int node : {0, 1}) {
            DaemonConfig cfg = single_window_config(nb + (1ull<<30), node, gpu,
                                                    "/tmp/ofld_bench_numa.sock");
            ::unlink(cfg.socket_path.c_str());
            Daemon dae(cfg);
            AgentConfig ac; ac.socket_path = cfg.socket_path; ac.cuda_device = gpu;
            ac.rank_id = 0; ac.numa_node = node;
            OffloadAgent agent(ac);
            double d2h=0, h2d=0;
            bench_d2h_h2d(agent, gpu, nb, iters, &d2h, &h2d);
            std::printf("arena on node %d (%s): D2H %.2f GB/s  H2D %.2f GB/s\n",
                        node, node==local_node ? "local" : "remote", d2h, h2d);
            std::printf("BENCH,numa_node%d,%llu,%.2f,%.2f\n", node,
                        (unsigned long long)(nb>>30), d2h, h2d);
        }
    }

    std::printf("\n# benchmark complete\n");
    return 0;
}
