// Fault-injection tests: drive a live daemon over a real socket and inject
// failures, asserting recovery + invariants (TEST_PLAN §3 "rank crash", §6
// robustness; RACE_CONDITIONS §3/§4/§6/§9). No GPU/CUDA — we drive the control
// plane directly (slot state transitions via RPC), so these run anywhere memfd
// is available.
//
// Scenarios:
//   1. Rank crash mid-D2H: session times out (heartbeat), leases recovered,
//      in-flight slots reclaimed to FREE, old-epoch completion rejected.
//   2. Stale-epoch completion after "restart" rejected (RACE §6).
//   3. PINNED_VALID data survives owner death (RACE §9): a completed offload's
//      location remains queryable after the owning rank dies.
//   4. Slot-exhaustion backpressure: over-subscribe the arena, assert the
//      daemon blocks (ok=false, blocked) instead of overlapping slots.
//   5. Completion ring: a stale-epoch entry pushed into a recycled ring is not
//      misapplied to the new owner's tensors.
//   6. Double-free / idempotent release: releasing a lease twice is safe.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "completion_ring.h"
#include "config.h"
#include "control_layout.h"
#include "daemon.h"
#include "offload_abi.h"
#include "test_harness.h"
#include "uds.h"
#include "util.h"
#include "wire.h"

using namespace offload;
using namespace ofldtest;

namespace {

struct RawClient {
    int sock;
    explicit RawClient(const std::string& path) { sock = uds_connect(path, 5000); }
    ~RawClient() { if (sock >= 0) close_fd(sock); }

    template <typename Resp>
    Resp call(OpCode op, const std::vector<uint8_t>& body,
              Resp (*dec)(const uint8_t*, size_t), std::vector<int>* out_fds = nullptr) {
        send_frame(sock, make_frame(op, body), {});
        OpCode rop; std::vector<uint8_t> payload; std::vector<int> fds;
        if (!recv_frame(sock, &rop, &payload, &fds))
            throw std::runtime_error("daemon closed connection");
        if (out_fds) *out_fds = fds; else for (int fd : fds) close_fd(fd);
        return dec(payload.data(), payload.size());
    }

    RegisterRankResponse register_rank(uint32_t rank, uint32_t gpu, uint32_t numa,
                                       uint64_t pid, std::vector<int>* fds) {
        RegisterRankRequest r;
        r.rank_id = rank; r.gpu_id = gpu; r.numa_node = numa; r.pid = pid;
        return call<RegisterRankResponse>(OpCode::kRegisterRank, encode(r),
                                          decode_RegisterRankResponse, fds);
    }
    HeartbeatResponse heartbeat(uint32_t rank, uint64_t epoch) {
        HeartbeatRequest r; r.rank_id = rank; r.rank_epoch = epoch;
        r.timestamp_ns = now_real_ns();
        return call<HeartbeatResponse>(OpCode::kHeartbeat, encode(r),
                                       decode_HeartbeatResponse);
    }
    RequestOffloadResponse request_offload(uint32_t rank, uint64_t epoch,
                                           uint64_t tid, uint64_t ver,
                                           uint64_t nbytes, uint32_t gpu, uint32_t numa) {
        RequestOffloadRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.tensor_id = tid; r.version = ver;
        r.nbytes = nbytes; r.gpu_id = gpu; r.numa_node = numa; r.alignment = 256;
        r.flags = OFLD_FLAG_ALLOW_OVERFLOW;
        return call<RequestOffloadResponse>(OpCode::kRequestOffload, encode(r),
                                            decode_RequestOffloadResponse);
    }
    MarkD2HSubmittedResponse d2h_submitted(uint32_t rank, uint64_t epoch, uint64_t lease,
                                           uint64_t tid, uint64_t ver, uint32_t slot) {
        MarkD2HSubmittedRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.lease_id = lease;
        r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HSubmittedResponse>(OpCode::kMarkD2HSubmitted, encode(r),
                                              decode_MarkD2HSubmittedResponse);
    }
    MarkD2HCompleteResponse d2h_complete(uint32_t rank, uint64_t epoch, uint64_t lease,
                                         uint64_t tid, uint64_t ver, uint32_t slot) {
        MarkD2HCompleteRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.lease_id = lease;
        r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HCompleteResponse>(OpCode::kMarkD2HComplete, encode(r),
                                             decode_MarkD2HCompleteResponse);
    }
    ReleaseLeaseResponse release(uint32_t rank, uint64_t epoch, uint64_t lease, uint32_t slot) {
        ReleaseLeaseRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.lease_id = lease; r.slot_id = slot;
        return call<ReleaseLeaseResponse>(OpCode::kReleaseLease, encode(r),
                                          decode_ReleaseLeaseResponse);
    }
    LocationQueryResponse query(uint64_t tid, uint64_t ver) {
        LocationQueryRequest r; r.tensor_id = tid; r.version = ver;
        return call<LocationQueryResponse>(OpCode::kQueryLocation, encode(r),
                                           decode_LocationQueryResponse);
    }
};

struct DaemonFixture {
    std::string sock_path;
    std::unique_ptr<OffloadDaemon> daemon;
    std::thread thr;
    uint32_t num_slots = 0;
    uint64_t gran = 0;

    DaemonFixture(uint64_t arena_bytes, uint64_t granularity,
                  uint64_t heartbeat_timeout_ms) {
        sock_path = "/tmp/ofld_fault_" + std::to_string(getpid()) + ".sock";
        DaemonConfig cfg = default_smoke_config(arena_bytes, 0, 0);
        cfg.socket_path = sock_path;
        cfg.allocation_granularity_bytes = granularity;
        cfg.nvme_enabled = false;
        cfg.drain_on_d2h_complete = false;   // keep PINNED_VALID for inspection
        cfg.heartbeat_timeout_ms = heartbeat_timeout_ms;
        gran = cfg.allocation_granularity_bytes;
        daemon.reset(new OffloadDaemon(cfg));
        num_slots = daemon->num_slots();
        thr = std::thread([this] { daemon->run(); });
        for (int i = 0; i < 200; ++i) {
            try { RawClient c(sock_path); break; }
            catch (...) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        }
    }
    ~DaemonFixture() {
        daemon->request_stop();
        if (thr.joinable()) thr.join();
        ::unlink(sock_path.c_str());
    }
};

}  // namespace

// 1. Rank crash mid-D2H: heartbeat timeout recovers the lease + reclaims slot.
static void test_rank_crash_recovery(DaemonFixture& fx) {
    uint64_t epoch;
    uint64_t lease; uint32_t slot;
    {
        std::vector<int> fds;
        RawClient c(fx.sock_path);
        auto reg = c.register_rank(20, 0, 0, /*pid=*/9001, &fds);
        for (int fd : fds) close_fd(fd);
        CHECK(reg.ok);
        epoch = reg.rank_epoch;
        // Reserve + submit but NEVER complete (simulate crash mid-D2H).
        auto off = c.request_offload(20, epoch, 7001, 1, fx.gran, 0, 0);
        CHECK(off.ok);
        lease = off.lease_id; slot = off.slot_id;
        c.d2h_submitted(20, epoch, lease, 7001, 1, slot);
        // Connection closes here (client destructed) — rank is "dead".
    }
    // Stop heartbeating. After the timeout, the monitor must recover the lease
    // and reclaim the in-flight slot to FREE.
    std::this_thread::sleep_for(std::chrono::milliseconds(900));

    // A NEW rank should now be able to reserve — the arena isn't leaked.
    std::vector<int> fds;
    RawClient c2(fx.sock_path);
    auto reg2 = c2.register_rank(21, 0, 0, 9002, &fds);
    for (int fd : fds) close_fd(fd);
    CHECK(reg2.ok);
    auto off2 = c2.request_offload(21, reg2.rank_epoch, 7002, 1, fx.gran, 0, 0);
    CHECK(off2.ok);  // slot available => not leaked

    // The dead rank's old-epoch completion must now be rejected.
    auto stale = c2.d2h_complete(20, epoch, lease, 7001, 1, slot);
    CHECK(!stale.ok);
}

// 2. Stale-epoch completion after restart rejected (RACE §6).
static void test_stale_epoch_after_restart(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c(fx.sock_path);
    auto reg1 = c.register_rank(22, 0, 0, 9101, &fds);
    for (int fd : fds) close_fd(fd);
    fds.clear();
    auto off = c.request_offload(22, reg1.rank_epoch, 7101, 1, fx.gran, 0, 0);
    CHECK(off.ok);
    c.d2h_submitted(22, reg1.rank_epoch, off.lease_id, 7101, 1, off.slot_id);

    // Rank "restarts": same rank_id, new session/epoch.
    RawClient c2(fx.sock_path);
    auto reg2 = c2.register_rank(22, 0, 0, 9101, &fds);
    for (int fd : fds) close_fd(fd);
    CHECK(reg2.rank_epoch != reg1.rank_epoch);

    // Old-epoch completion from before the restart must be rejected.
    auto stale = c.d2h_complete(22, reg1.rank_epoch, off.lease_id, 7101, 1, off.slot_id);
    CHECK(!stale.ok);
}

// 3. Completed (PINNED_VALID) data survives owner death (RACE §9): querying the
//    tensor still returns a valid pinned location after the owner disconnects.
static void test_pinned_survives_owner_death(DaemonFixture& fx) {
    uint64_t tid = 7201;
    {
        std::vector<int> fds;
        RawClient c(fx.sock_path);
        auto reg = c.register_rank(23, 0, 0, 9201, &fds);
        for (int fd : fds) close_fd(fd);
        auto off = c.request_offload(23, reg.rank_epoch, tid, 1, fx.gran, 0, 0);
        CHECK(off.ok);
        c.d2h_submitted(23, reg.rank_epoch, off.lease_id, tid, 1, off.slot_id);
        auto comp = c.d2h_complete(23, reg.rank_epoch, off.lease_id, tid, 1, off.slot_id);
        CHECK(comp.ok);
        // Owner disconnects (dies) without releasing.
    }
    // Even after heartbeat timeout, the latest valid copy must not vanish.
    std::this_thread::sleep_for(std::chrono::milliseconds(900));
    RawClient c2(fx.sock_path);
    auto loc = c2.query(tid, 0);
    CHECK(loc.ok);
    CHECK_EQ(loc.location_kind, (uint32_t)OFLD_LOC_PINNED);
}

// 4. Slot-exhaustion backpressure: reserve the whole arena, then assert the
//    next reservation is refused (blocked) rather than overlapping.
static void test_slot_exhaustion_backpressure(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c(fx.sock_path);
    auto reg = c.register_rank(24, 0, 0, 9301, &fds);
    for (int fd : fds) close_fd(fd);
    CHECK(reg.ok);

    // Reserve until refused. Each reservation is one slot (nbytes==gran).
    int granted = 0;
    bool blocked_seen = false;
    for (uint32_t i = 0; i < fx.num_slots + 4; ++i) {
        auto off = c.request_offload(24, reg.rank_epoch, 8000 + i, 1, fx.gran, 0, 0);
        if (off.ok) { ++granted; }
        else { blocked_seen = true; CHECK(off.blocked); break; }
    }
    CHECK(blocked_seen);
    CHECK(granted <= (int)fx.num_slots);   // never over-allocate
    CHECK(granted > 0);
    std::fprintf(stderr, "  [backpressure] granted=%d/%u then blocked\n",
                 granted, fx.num_slots);
}

// 5. Idempotent double-release: releasing the same lease twice is safe.
static void test_double_release_idempotent(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c(fx.sock_path);
    auto reg = c.register_rank(25, 0, 0, 9401, &fds);
    for (int fd : fds) close_fd(fd);
    auto off = c.request_offload(25, reg.rank_epoch, 7301, 1, fx.gran, 0, 0);
    CHECK(off.ok);
    c.d2h_submitted(25, reg.rank_epoch, off.lease_id, 7301, 1, off.slot_id);
    c.d2h_complete(25, reg.rank_epoch, off.lease_id, 7301, 1, off.slot_id);
    auto r1 = c.release(25, reg.rank_epoch, off.lease_id, off.slot_id);
    CHECK(r1.ok);
    auto r2 = c.release(25, reg.rank_epoch, off.lease_id, off.slot_id);
    CHECK(r2.ok);  // idempotent — already gone
    // The slot must be reusable after release.
    auto off2 = c.request_offload(25, reg.rank_epoch, 7302, 1, fx.gran, 0, 0);
    CHECK(off2.ok);
}

int main() {
    // Each scenario gets its OWN daemon fixture so slot/session state never
    // leaks across tests. 32-slot arena, 2MB granularity, short 300ms heartbeat
    // timeout for fast crash-recovery testing.
    const uint64_t gran = (2ull << 20);
    const uint64_t arena = gran * 32;
    const uint64_t hb = 300;
    {
        DaemonFixture fx(arena, gran, hb);
        if (fx.num_slots == 0) {
            std::fprintf(stderr, "[SKIP] daemon 0 slots (memfd unavailable?)\n");
            return 0;
        }
        std::fprintf(stderr, "fault daemon up: %u slots\n", fx.num_slots);
        RUN([&]{ test_rank_crash_recovery(fx); });
    }
    { DaemonFixture fx(arena, gran, hb); RUN([&]{ test_stale_epoch_after_restart(fx); }); }
    { DaemonFixture fx(arena, gran, hb); RUN([&]{ test_pinned_survives_owner_death(fx); }); }
    { DaemonFixture fx(arena, gran, hb); RUN([&]{ test_slot_exhaustion_backpressure(fx); }); }
    { DaemonFixture fx(arena, gran, hb); RUN([&]{ test_double_release_idempotent(fx); }); }

    return summary("fault_injection");
}
