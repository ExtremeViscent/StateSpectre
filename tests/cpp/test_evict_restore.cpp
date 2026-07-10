// Functional GPU test: full destructive evict + restore byte-correctness via
// the real rank agent and a live daemon (TEST_PLAN §2).
//
//   1. allocate a CUDA buffer, fill with a deterministic pattern
//   2. evict destructively (D2H into a pinned slot; original storage freed via
//      the invalidate callback)
//   3. restore into a fresh CUDA buffer
//   4. copy back to host and assert byte-identical to the original pattern
//   5. repeat many times to prove one-off registration + no slot leak
//
// Requires a GPU and memfd_create. Run outside a restrictive sandbox.

#include <cuda_runtime.h>
#include <unistd.h>

#include <atomic>
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
#include "test_harness.h"
#include "util.h"

using namespace offload;
using namespace ofldtest;

#define CU(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,        \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

namespace {

struct DaemonFixture {
    std::string sock_path;
    std::unique_ptr<OffloadDaemon> daemon;
    std::thread thr;
    uint32_t num_slots = 0;

    DaemonFixture(uint64_t arena_bytes, uint64_t gran, int numa, int gpu,
                  bool drain_on_complete) {
        sock_path = "/tmp/ofld_gpu_test_" + std::to_string(getpid()) + ".sock";
        DaemonConfig cfg = default_smoke_config(arena_bytes, numa, gpu);
        cfg.socket_path = sock_path;
        cfg.allocation_granularity_bytes = gran;
        cfg.nvme_enabled = false;
        cfg.drain_on_d2h_complete = drain_on_complete;
        daemon.reset(new OffloadDaemon(cfg));
        num_slots = daemon->num_slots();
        thr = std::thread([this] { daemon->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // NVMe variant: drain straight to NVMe using the given io engine.
    DaemonFixture(uint64_t arena_bytes, uint64_t gran, int numa, int gpu,
                  const std::string& nvme_path, const std::string& engine) {
        sock_path = "/tmp/ofld_gpu_test_" + std::to_string(getpid()) + ".sock";
        DaemonConfig cfg = default_smoke_config(arena_bytes, numa, gpu);
        cfg.socket_path = sock_path;
        cfg.allocation_granularity_bytes = gran;
        cfg.nvme_enabled = true;
        cfg.nvme_path = nvme_path;
        cfg.nvme_io_engine = engine;
        cfg.nvme_direct_io = true;
        cfg.nvme_stripe = true;
        cfg.drain_target = DrainTarget::kNvmeOnly;
        cfg.drain_on_d2h_complete = true;
        daemon.reset(new OffloadDaemon(cfg));
        num_slots = daemon->num_slots();
        thr = std::thread([this] { daemon->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ~DaemonFixture() {
        daemon->request_stop();
        if (thr.joinable()) thr.join();
        ::unlink(sock_path.c_str());
    }
};

// Fill a device buffer with a deterministic byte pattern derived from seed.
void fill_pattern(void* dptr, size_t n, uint32_t seed) {
    std::vector<uint8_t> h(n);
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1103515245u + 12345u;
        h[i] = (uint8_t)(x >> 16);
    }
    CU(cudaMemcpy(dptr, h.data(), n, cudaMemcpyHostToDevice));
}

bool verify_pattern(const void* dptr, size_t n, uint32_t seed) {
    std::vector<uint8_t> h(n), expect(n);
    CU(cudaMemcpy(h.data(), dptr, n, cudaMemcpyDeviceToHost));
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1103515245u + 12345u;
        expect[i] = (uint8_t)(x >> 16);
    }
    return std::memcmp(h.data(), expect.data(), n) == 0;
}

}  // namespace

// Single destructive evict + restore, byte-identical.
static void test_evict_restore_bytes(OffloadAgent& agent) {
    const size_t N = 4ull << 20;  // 4 MB
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, N));
    fill_pattern(dptr, N, 0xC0FFEE);

    // Simulate "storage replacement": our invalidate callback frees the buffer.
    std::atomic<bool> invalidated{false};
    void* to_free = dptr;
    InvalidateCallback cb = [&](uint64_t cookie) {
        (void)cookie;
        cudaFree(to_free);       // real VRAM release after D2H completes
        invalidated.store(true);
    };

    TensorMeta meta;
    meta.tensor_id = 7777; meta.version = 1; meta.nbytes = N; meta.name = "t";

    OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, /*destructive=*/true,
                                   meta, cb, /*cookie=*/7777, /*wait=*/true);
    CHECK(tk.ok);
    CHECK(invalidated.load());  // destructive invalidation ran (wait=true)

    // Restore into a fresh buffer.
    void* rptr = nullptr;
    CU(cudaMalloc(&rptr, N));
    RestoreResult rr = agent.restore(7777, 1, (uint64_t)rptr, N, 0, /*wait=*/true);
    CHECK(rr.ok);
    CHECK(verify_pattern(rptr, N, 0xC0FFEE));  // byte-identical
    CU(cudaFree(rptr));
}

// Repeated reuse: prove one-off registration + no slot leak over many cycles.
static void test_repeated_reuse(OffloadAgent& agent) {
    const size_t N = 2ull << 20;  // 2 MB = one slot
    const int kIters = 200;
    bool all_ok = true;
    for (int i = 0; i < kIters; ++i) {
        void* dptr = nullptr;
        CU(cudaMalloc(&dptr, N));
        fill_pattern(dptr, N, (uint32_t)(i + 1));
        void* to_free = dptr;
        InvalidateCallback cb = [to_free](uint64_t) { cudaFree(to_free); };
        TensorMeta meta;
        meta.tensor_id = 20000 + i; meta.version = 1; meta.nbytes = N;
        OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, true, meta, cb,
                                       20000 + i, true);
        if (!tk.ok) { all_ok = false; break; }
        void* rptr = nullptr;
        CU(cudaMalloc(&rptr, N));
        RestoreResult rr = agent.restore(20000 + i, 1, (uint64_t)rptr, N, 0, true);
        if (!rr.ok || !verify_pattern(rptr, N, (uint32_t)(i + 1))) {
            all_ok = false; cudaFree(rptr); break;
        }
        // Release the slot so it recycles (proves no leak across 200 iters in a
        // 64-slot arena).
        agent.discard(20000 + i, 1);
        cudaFree(rptr);
    }
    CHECK(all_ok);
    std::fprintf(stderr, "  [repeated-reuse] %d evict/restore cycles ok\n", kIters);
}

// Non-destructive copy: original remains valid.
static void test_copy_nondestructive(OffloadAgent& agent) {
    const size_t N = 1ull << 20;
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, N));
    fill_pattern(dptr, N, 0xABCDE);
    TensorMeta meta; meta.tensor_id = 30001; meta.version = 1; meta.nbytes = N;
    OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, /*destructive=*/false,
                                   meta, nullptr, 0, true);
    CHECK(tk.ok);
    // Original must still be valid (non-destructive).
    CHECK(verify_pattern(dptr, N, 0xABCDE));
    void* rptr = nullptr;
    CU(cudaMalloc(&rptr, N));
    RestoreResult rr = agent.restore(30001, 1, (uint64_t)rptr, N, 0, true);
    CHECK(rr.ok);
    CHECK(verify_pattern(rptr, N, 0xABCDE));
    CU(cudaFree(dptr));
    CU(cudaFree(rptr));
}

// Regression: a tensor LARGER than the registration chunk / spanning many
// slots must evict+restore byte-identically. A single cudaMemcpyAsync target
// must lie within one registered host range; this catches the class of bug
// where chunked registration makes a boundary-crossing copy fail/corrupt.
static void test_large_tensor(OffloadAgent& agent) {
    const size_t N = 768ull << 20;  // 768 MB: crosses the 512 MB reg chunk
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, N));
    fill_pattern(dptr, N, 0xBADC0DE);
    void* to_free = dptr;
    InvalidateCallback cb = [to_free](uint64_t) { cudaFree(to_free); };
    TensorMeta meta; meta.tensor_id = 31001; meta.version = 1; meta.nbytes = N;
    OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, /*destructive=*/true,
                                   meta, cb, 31001, true);
    CHECK(tk.ok);
    void* rptr = nullptr;
    CU(cudaMalloc(&rptr, N));
    RestoreResult rr = agent.restore(31001, 1, (uint64_t)rptr, N, 0, true);
    CHECK(rr.ok);
    CHECK(verify_pattern(rptr, N, 0xBADC0DE));  // byte-identical across boundary
    agent.discard(31001, 1);
    CU(cudaFree(rptr));
}
// Drain path: with drain_on_d2h_complete, tensor migrates PINNED->PAGEABLE and
// restore must still reconstruct it byte-perfectly (readback path).
static void test_restore_after_drain(const std::string& sock, int gpu, int numa) {
    AgentConfig acfg;
    acfg.socket_path = sock; acfg.cuda_device = gpu; acfg.rank_id = 1;
    acfg.numa_node = numa;
    OffloadAgent agent(acfg);

    const size_t N = 3ull << 20;
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, N));
    fill_pattern(dptr, N, 0x5A5A5A);
    void* to_free = dptr;
    InvalidateCallback cb = [to_free](uint64_t) { cudaFree(to_free); };
    TensorMeta meta; meta.tensor_id = 40001; meta.version = 1; meta.nbytes = N;
    OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, true, meta, cb, 40001, true);
    CHECK(tk.ok);

    // Give the drain worker time to migrate pinned -> pageable and recycle slot.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::string loc = agent.location(40001, 1);
    std::fprintf(stderr, "  [drain] post-drain location=%s\n", loc.c_str());

    void* rptr = nullptr;
    CU(cudaMalloc(&rptr, N));
    RestoreResult rr = agent.restore(40001, 1, (uint64_t)rptr, N, 0, true);
    CHECK(rr.ok);
    CHECK(verify_pattern(rptr, N, 0x5A5A5A));  // byte-identical after readback
    CU(cudaFree(rptr));
}

// NVMe spill: tensor drains GPU->pinned->NVMe (via the given io engine), the
// pinned slot recycles, and restore reads NVMe->pinned->GPU byte-perfectly.
static void test_nvme_roundtrip(const std::string& sock, int gpu, int numa,
                                const char* engine_label) {
    AgentConfig acfg;
    acfg.socket_path = sock; acfg.cuda_device = gpu; acfg.rank_id = 2;
    acfg.numa_node = numa;
    OffloadAgent agent(acfg);

    const size_t N = 5ull << 20;  // 5 MB, spans multiple 2MB slots + stripes
    void* dptr = nullptr;
    CU(cudaMalloc(&dptr, N));
    fill_pattern(dptr, N, 0x9E3779B9);
    void* to_free = dptr;
    InvalidateCallback cb = [to_free](uint64_t) { cudaFree(to_free); };
    TensorMeta meta; meta.tensor_id = 50001; meta.version = 1; meta.nbytes = N;
    OffloadTicket tk = agent.evict((uint64_t)dptr, N, 0, true, meta, cb, 50001, true);
    CHECK(tk.ok);

    // Wait for the drain worker to write NVMe and recycle the pinned slot.
    std::string loc;
    for (int i = 0; i < 100; ++i) {
        loc = agent.location(50001, 1);
        if (loc == "nvme") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::fprintf(stderr, "  [nvme:%s] post-drain location=%s\n", engine_label,
                 loc.c_str());
    CHECK_STREQ(loc, "nvme");

    void* rptr = nullptr;
    CU(cudaMalloc(&rptr, N));
    RestoreResult rr = agent.restore(50001, 1, (uint64_t)rptr, N, 0, true);
    if (!rr.ok) std::fprintf(stderr, "  [nvme:%s] restore FAILED: %s\n",
                             engine_label, rr.message.c_str());
    CHECK(rr.ok);
    CHECK(verify_pattern(rptr, N, 0x9E3779B9));  // byte-identical after NVMe read
    CU(cudaFree(rptr));
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "[SKIP] no CUDA device available\n");
        return 0;
    }
    const int gpu = 0;
    CU(cudaSetDevice(gpu));
    int numa = 0;  // smoke config uses node 0; agent auto-detects but we pin.

    const uint64_t gran = (2ull << 20);
    const uint64_t arena = gran * 64;  // 128 MB, 64 slots

    // --- Suite A: no auto-drain, slots stay PINNED_VALID ---
    {
        DaemonFixture fx(arena, gran, numa, gpu, /*drain_on_complete=*/false);
        if (fx.num_slots == 0) {
            std::fprintf(stderr, "[SKIP] daemon 0 slots (memfd unavailable?)\n");
            return 0;
        }
        std::fprintf(stderr, "daemon up (no-drain): %u slots\n", fx.num_slots);
        AgentConfig acfg;
        acfg.socket_path = fx.sock_path; acfg.cuda_device = gpu; acfg.rank_id = 0;
        acfg.numa_node = numa;
        OffloadAgent agent(acfg);

        RUN([&]{ test_evict_restore_bytes(agent); });
        RUN([&]{ test_repeated_reuse(agent); });
        RUN([&]{ test_copy_nondestructive(agent); });
        agent.assert_no_inflight();
    }

    // --- Suite A2: large tensor crossing the registration boundary ---
    {
        const uint64_t big_gran = (64ull << 20);
        const uint64_t big_arena = (uint64_t)(4ull << 30);  // 4 GiB, 64 slots
        DaemonFixture fx(big_arena, big_gran, numa, gpu, /*drain_on_complete=*/false);
        std::fprintf(stderr, "daemon up (large): %u slots x %lluMB\n",
                     fx.num_slots, (unsigned long long)(big_gran >> 20));
        AgentConfig acfg;
        acfg.socket_path = fx.sock_path; acfg.cuda_device = gpu; acfg.rank_id = 0;
        acfg.numa_node = numa;
        OffloadAgent agent(acfg);
        RUN([&]{ test_large_tensor(agent); });
        agent.assert_no_inflight();
    }

    // --- Suite B: auto-drain to pageable, restore via readback ---
    {
        DaemonFixture fx(arena, gran, numa, gpu, /*drain_on_complete=*/true);
        std::fprintf(stderr, "daemon up (drain): %u slots\n", fx.num_slots);
        RUN([&]{ test_restore_after_drain(fx.sock_path, gpu, numa); });
    }

    // --- Suite C: NVMe spill, both io engines (pwrite + io_uring) ---
    {
        std::string ndir = "/tmp/ofld_nvme_" + std::to_string(getpid()) + "_pw";
        DaemonFixture fx(arena, gran, numa, gpu, ndir, "pwrite");
        std::fprintf(stderr, "daemon up (nvme pwrite): %u slots\n", fx.num_slots);
        RUN([&]{ test_nvme_roundtrip(fx.sock_path, gpu, numa, "pwrite"); });
    }
    {
        std::string ndir = "/tmp/ofld_nvme_" + std::to_string(getpid()) + "_ur";
        DaemonFixture fx(arena, gran, numa, gpu, ndir, "io_uring");
        std::fprintf(stderr, "daemon up (nvme io_uring): %u slots\n", fx.num_slots);
        RUN([&]{ test_nvme_roundtrip(fx.sock_path, gpu, numa, "io_uring"); });
    }

    return summary("evict_restore");
}
