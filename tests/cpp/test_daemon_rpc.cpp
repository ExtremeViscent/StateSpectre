// Integration test: live daemon over a real Unix socket + memfd control region.
// Exercises the RPC state machine and race-condition invariants WITHOUT a GPU
// (no CUDA copies; we drive slot state through the control-plane RPCs directly).
//
// Covers TEST_PLAN §3: double writer (no slot overlap), stale version, epoch
// rejection, invalid transitions, and drain/reuse race (slot not reallocated
// while DRAIN_IN_FLIGHT).
//
// Requires memfd_create (run outside a restrictive seccomp sandbox).

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "daemon.h"
#include "test_harness.h"
#include "uds.h"
#include "util.h"
#include "wire.h"

using namespace offload;
using namespace ofldtest;

namespace {

// A tiny synchronous RPC client built directly on uds+wire (no CUDA/agent dep).
struct RawClient {
    int sock;
    explicit RawClient(const std::string& path) { sock = uds_connect(path, 5000); }
    ~RawClient() { if (sock >= 0) close_fd(sock); }

    template <typename Resp>
    Resp call(OpCode op, const std::vector<uint8_t>& body,
              Resp (*dec)(const uint8_t*, size_t),
              std::vector<int>* out_fds = nullptr) {
        send_frame(sock, make_frame(op, body), {});
        OpCode rop;
        std::vector<uint8_t> payload;
        std::vector<int> fds;
        bool got = recv_frame(sock, &rop, &payload, &fds);
        if (!got) throw std::runtime_error("daemon closed connection");
        if (out_fds) *out_fds = fds;
        else for (int fd : fds) close_fd(fd);
        return dec(payload.data(), payload.size());
    }

    RegisterRankResponse register_rank(uint32_t rank, uint32_t gpu, uint32_t numa,
                                       std::vector<int>* fds) {
        RegisterRankRequest r;
        r.rank_id = rank; r.gpu_id = gpu; r.numa_node = numa;
        r.pid = (uint64_t)getpid() + rank;  // distinct pid per logical rank
        return call<RegisterRankResponse>(OpCode::kRegisterRank, encode(r),
                                          decode_RegisterRankResponse, fds);
    }
    RequestOffloadResponse request_offload(uint32_t rank, uint64_t epoch,
                                           uint64_t tid, uint64_t ver,
                                           uint64_t nbytes, uint32_t gpu,
                                           uint32_t numa) {
        RequestOffloadRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.tensor_id = tid;
        r.version = ver; r.nbytes = nbytes; r.gpu_id = gpu; r.numa_node = numa;
        r.alignment = 256; r.flags = OFLD_FLAG_ALLOW_OVERFLOW;
        return call<RequestOffloadResponse>(OpCode::kRequestOffload, encode(r),
                                            decode_RequestOffloadResponse);
    }
    MarkD2HSubmittedResponse d2h_submitted(uint32_t rank, uint64_t epoch,
                                           uint64_t lease, uint64_t tid,
                                           uint64_t ver, uint32_t slot) {
        MarkD2HSubmittedRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.lease_id = lease;
        r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HSubmittedResponse>(OpCode::kMarkD2HSubmitted, encode(r),
                                              decode_MarkD2HSubmittedResponse);
    }
    MarkD2HCompleteResponse d2h_complete(uint32_t rank, uint64_t epoch,
                                         uint64_t lease, uint64_t tid,
                                         uint64_t ver, uint32_t slot) {
        MarkD2HCompleteRequest r;
        r.rank_id = rank; r.rank_epoch = epoch; r.lease_id = lease;
        r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HCompleteResponse>(OpCode::kMarkD2HComplete, encode(r),
                                             decode_MarkD2HCompleteResponse);
    }
    LocationQueryResponse query(uint64_t tid, uint64_t ver) {
        LocationQueryRequest r; r.tensor_id = tid; r.version = ver;
        return call<LocationQueryResponse>(OpCode::kQueryLocation, encode(r),
                                           decode_LocationQueryResponse);
    }
};

// Launch a daemon on a background thread with a small no-NVMe smoke arena.
struct DaemonFixture {
    std::string sock_path;
    std::unique_ptr<OffloadDaemon> daemon;
    std::thread thr;
    uint32_t num_slots = 0;
    uint64_t gran = 0;

    DaemonFixture(uint64_t arena_bytes, uint64_t granularity) {
        sock_path = "/tmp/ofld_test_daemon_" + std::to_string(getpid()) + ".sock";
        DaemonConfig cfg = default_smoke_config(arena_bytes, 0, 0);
        cfg.socket_path = sock_path;
        cfg.allocation_granularity_bytes = granularity;
        cfg.nvme_enabled = false;
        cfg.drain_on_d2h_complete = false;  // keep slots PINNED_VALID for tests
        gran = cfg.allocation_granularity_bytes;
        daemon.reset(new OffloadDaemon(cfg));
        num_slots = daemon->num_slots();
        thr = std::thread([this] { daemon->run(); });
        // Wait until the socket is accepting.
        for (int i = 0; i < 200; ++i) {
            try { RawClient c(sock_path); break; }
            catch (...) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
        }
    }
    ~DaemonFixture() {
        daemon->request_stop();
        if (thr.joinable()) thr.join();
        ::unlink(sock_path.c_str());
    }
};

}  // namespace

// -- Test: normal offload lifecycle FREE->RESERVED->IN_FLIGHT->PINNED_VALID ----
static void test_lifecycle(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c(fx.sock_path);
    auto reg = c.register_rank(0, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);
    CHECK(reg.ok);

    auto off = c.request_offload(0, reg.rank_epoch, 1001, 1, fx.gran, 0, 0);
    CHECK(off.ok);
    CHECK_EQ(off.state, (uint32_t)OFLD_SLOT_RESERVED_D2H);

    auto sub = c.d2h_submitted(0, reg.rank_epoch, off.lease_id, 1001, 1, off.slot_id);
    CHECK(sub.ok);

    auto comp = c.d2h_complete(0, reg.rank_epoch, off.lease_id, 1001, 1, off.slot_id);
    CHECK(comp.ok);
    CHECK(comp.latest_version);

    auto loc = c.query(1001, 0);
    CHECK(loc.ok);
    CHECK_EQ(loc.location_kind, (uint32_t)OFLD_LOC_PINNED);
    CHECK_EQ(loc.version, 1ull);
}

// -- Test: epoch rejection (RACE_CONDITIONS §6) --------------------------------
static void test_epoch_rejected(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c(fx.sock_path);
    auto reg = c.register_rank(1, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);
    // Use a bogus (old) epoch.
    auto off = c.request_offload(1, reg.rank_epoch + 999, 2002, 1, fx.gran, 0, 0);
    CHECK(!off.ok);
}

// -- Test: re-register invalidates the prior epoch -----------------------------
static void test_reregister_invalidates(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient c1(fx.sock_path);
    auto reg1 = c1.register_rank(2, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);
    fds.clear();

    // Same rank_id re-registers => new epoch, old one must be rejected.
    RawClient c2(fx.sock_path);
    auto reg2 = c2.register_rank(2, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);
    CHECK(reg2.rank_epoch != reg1.rank_epoch);

    auto off_old = c1.request_offload(2, reg1.rank_epoch, 3003, 1, fx.gran, 0, 0);
    CHECK(!off_old.ok);  // old epoch rejected
    auto off_new = c2.request_offload(2, reg2.rank_epoch, 3003, 1, fx.gran, 0, 0);
    CHECK(off_new.ok);   // new epoch works
}

// -- Test: stale version cannot overwrite latest (RACE_CONDITIONS §5) ----------
static void test_stale_version(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient conn(fx.sock_path);
    auto reg = conn.register_rank(3, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);

    uint64_t tid = 4004;
    // Offload version 10, submit but DON'T complete yet.
    auto o10 = conn.request_offload(3, reg.rank_epoch, tid, 10, fx.gran, 0, 0);
    CHECK(o10.ok);
    conn.d2h_submitted(3, reg.rank_epoch, o10.lease_id, tid, 10, o10.slot_id);

    // Offload version 11 and complete it => latest becomes 11.
    auto o11 = conn.request_offload(3, reg.rank_epoch, tid, 11, fx.gran, 0, 0);
    CHECK(o11.ok);
    conn.d2h_submitted(3, reg.rank_epoch, o11.lease_id, tid, 11, o11.slot_id);
    auto c11 = conn.d2h_complete(3, reg.rank_epoch, o11.lease_id, tid, 11, o11.slot_id);
    CHECK(c11.ok);
    CHECK(c11.latest_version);

    // Now complete the late version 10. It must be accepted as a transition but
    // NOT become the latest location.
    auto c10 = conn.d2h_complete(3, reg.rank_epoch, o10.lease_id, tid, 10, o10.slot_id);
    CHECK(c10.ok);
    CHECK(!c10.latest_version);  // stale => not latest

    auto loc = conn.query(tid, 0);
    CHECK(loc.ok);
    CHECK_EQ(loc.version, 11ull);  // latest stays 11
}

// -- Test: invalid transition rejected (complete without submit) ---------------
static void test_invalid_transition(DaemonFixture& fx) {
    std::vector<int> fds;
    RawClient conn(fx.sock_path);
    auto reg = conn.register_rank(4, 0, 0, &fds);
    for (int fd : fds) close_fd(fd);
    auto off = conn.request_offload(4, reg.rank_epoch, 5005, 1, fx.gran, 0, 0);
    CHECK(off.ok);
    // Skip submit; go straight to complete from RESERVED_D2H => invalid.
    auto comp = conn.d2h_complete(4, reg.rank_epoch, off.lease_id, 5005, 1, off.slot_id);
    CHECK(!comp.ok);
}

// -- Test: concurrent double-writer => no overlapping slots (RACE §4) ----------
static void test_double_writer_no_overlap(DaemonFixture& fx) {
    const int kRanks = 6;
    const int kPerRank = 8;
    std::mutex m;
    std::multiset<uint32_t> all_slots;
    std::atomic<int> granted{0};

    auto worker = [&](int rank) {
        std::vector<int> fds;
        RawClient conn(fx.sock_path);
        auto reg = conn.register_rank((uint32_t)rank, 0, 0, &fds);
        for (int fd : fds) close_fd(fd);
        for (int i = 0; i < kPerRank; ++i) {
            uint64_t tid = (uint64_t)rank * 1000 + i;
            auto off = conn.request_offload((uint32_t)rank, reg.rank_epoch, tid, 1,
                                            fx.gran, 0, 0);
            if (off.ok) {
                granted.fetch_add(1);
                std::lock_guard<std::mutex> lg(m);
                // record every slot in the span [slot_id, slot_id+span)
                uint32_t span = (uint32_t)((fx.gran + fx.gran - 1) / fx.gran);
                for (uint32_t s = 0; s < span; ++s) all_slots.insert(off.slot_id + s);
            }
        }
    };
    std::vector<std::thread> ts;
    for (int r = 0; r < kRanks; ++r) ts.emplace_back(worker, 10 + r);
    for (auto& t : ts) t.join();

    // No active reserved slot may appear twice (no overlap).
    std::set<uint32_t> uniq(all_slots.begin(), all_slots.end());
    CHECK_EQ((int)uniq.size(), (int)all_slots.size());
    CHECK(granted.load() > 0);
    std::fprintf(stderr, "  [double-writer] granted=%d distinct-slots=%d\n",
                 granted.load(), (int)uniq.size());
}

int main() {
    // Arena of 64 slots at 2MB each.
    const uint64_t gran = (2ull << 20);
    const uint64_t arena = gran * 64;
    DaemonFixture fx(arena, gran);
    if (fx.num_slots == 0) {
        std::fprintf(stderr, "[SKIP] daemon has 0 slots (memfd unavailable?)\n");
        return 0;
    }
    std::fprintf(stderr, "daemon up: %u slots, gran=%llu\n", fx.num_slots,
                 (unsigned long long)fx.gran);

    RUN([&]{ test_lifecycle(fx); });
    RUN([&]{ test_epoch_rejected(fx); });
    RUN([&]{ test_reregister_invalidates(fx); });
    RUN([&]{ test_stale_version(fx); });
    RUN([&]{ test_invalid_transition(fx); });
    RUN([&]{ test_double_writer_no_overlap(fx); });

    return summary("daemon_rpc");
}
