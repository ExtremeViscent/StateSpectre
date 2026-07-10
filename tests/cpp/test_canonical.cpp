// v2 canonical model-state integration tests: live daemon over a real Unix
// socket + memfd control region, no GPU (we drive slot state through the v1
// D2H opcodes against the daemon-synthesized canonical tid, exactly as the rank
// agent would). Covers TEST_PLAN_V2 §1-§6.
//
// Requires memfd_create (run outside a restrictive seccomp sandbox).

#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "daemon.h"
#include "test_harness.h"
#include "uds.h"
#include "util.h"
#include "wire.h"
#include "wire_v2.h"

using namespace offload;
using namespace ofldtest;

namespace {

struct RawClient {
    int sock;
    explicit RawClient(const std::string& path) { sock = uds_connect(path, 5000); }
    ~RawClient() { if (sock >= 0) close_fd(sock); }

    template <typename Resp>
    Resp call(OpCode op, const std::vector<uint8_t>& body,
              Resp (*dec)(const uint8_t*, size_t),
              std::vector<int>* out_fds = nullptr) {
        send_frame(sock, make_frame(op, body), {});
        OpCode rop; std::vector<uint8_t> payload; std::vector<int> fds;
        if (!recv_frame(sock, &rop, &payload, &fds))
            throw std::runtime_error("daemon closed connection");
        if (out_fds) *out_fds = fds; else for (int fd : fds) close_fd(fd);
        return dec(payload.data(), payload.size());
    }

    RegisterRankResponse register_rank(uint32_t rank, std::vector<int>* fds) {
        RegisterRankRequest r;
        r.rank_id = rank; r.gpu_id = 0; r.numa_node = 0;
        r.pid = (uint64_t)getpid() + rank;
        return call<RegisterRankResponse>(OpCode::kRegisterRank, encode(r),
                                          decode_RegisterRankResponse, fds);
    }
    RegisterJobResponse register_job(const std::string& name, uint64_t sched_id) {
        RegisterJobRequest r; r.tenant_id = 0; r.job_name = name;
        r.scheduler_job_id = sched_id;
        return call<RegisterJobResponse>(OpCode::kRegisterJob, encode(r),
                                         decode_RegisterJobResponse);
    }
    RequestCanonicalEvictResponse can_evict(const RequestCanonicalEvictRequest& r) {
        return call<RequestCanonicalEvictResponse>(OpCode::kRequestCanonicalEvict,
            encode(r), decode_RequestCanonicalEvictResponse);
    }
    MarkD2HSubmittedResponse d2h_sub(uint32_t rank, uint64_t epoch, uint64_t lease,
                                     uint64_t tid, uint64_t ver, uint32_t slot) {
        MarkD2HSubmittedRequest r; r.rank_id = rank; r.rank_epoch = epoch;
        r.lease_id = lease; r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HSubmittedResponse>(OpCode::kMarkD2HSubmitted, encode(r),
                                              decode_MarkD2HSubmittedResponse);
    }
    MarkD2HCompleteResponse d2h_done(uint32_t rank, uint64_t epoch, uint64_t lease,
                                     uint64_t tid, uint64_t ver, uint32_t slot) {
        MarkD2HCompleteRequest r; r.rank_id = rank; r.rank_epoch = epoch;
        r.lease_id = lease; r.tensor_id = tid; r.version = ver; r.slot_id = slot;
        return call<MarkD2HCompleteResponse>(OpCode::kMarkD2HComplete, encode(r),
                                             decode_MarkD2HCompleteResponse);
    }
    CommitCanonicalObjectResponse commit(uint32_t rank, uint64_t epoch,
                                         uint64_t oid, uint64_t lease, uint32_t slot,
                                         uint64_t hlo = 0, uint64_t hhi = 0) {
        CommitCanonicalObjectRequest r; r.rank_id = rank; r.rank_epoch = epoch;
        r.object_id = oid; r.lease_id = lease; r.slot_id = slot;
        r.content_hash_lo = hlo; r.content_hash_hi = hhi;
        return call<CommitCanonicalObjectResponse>(OpCode::kCommitCanonicalObject,
            encode(r), decode_CommitCanonicalObjectResponse);
    }
    SealModelVersionResponse seal(const JobKeyWire& job, uint32_t role,
                                  uint64_t ver, bool promote, bool fail_if_missing) {
        SealModelVersionRequest r; r.job = job; r.model_role = role;
        r.model_version = ver; r.promote = promote; r.fail_if_missing = fail_if_missing;
        return call<SealModelVersionResponse>(OpCode::kSealModelVersion, encode(r),
                                              decode_SealModelVersionResponse);
    }
    GetLatestSealedVersionResponse latest(const JobKeyWire& job, uint32_t role) {
        GetLatestSealedVersionRequest r; r.job = job; r.model_role = role;
        return call<GetLatestSealedVersionResponse>(OpCode::kGetLatestSealedVersion,
            encode(r), decode_GetLatestSealedVersionResponse);
    }
    GetManifestResponse manifest(const JobKeyWire& job, uint32_t role, uint64_t ver) {
        GetManifestRequest r; r.job = job; r.model_role = role; r.model_version = ver;
        return call<GetManifestResponse>(OpCode::kGetManifest, encode(r),
                                         decode_GetManifestResponse);
    }
    PullTensorResponse pull(const JobKeyWire& job, uint32_t role, uint64_t ver,
                            uint64_t oid, uint32_t transport, const std::string& dst) {
        PullTensorRequest r; r.job = job; r.model_role = role; r.model_version = ver;
        r.object_id = oid; r.transport = transport; r.target_descriptor = dst;
        return call<PullTensorResponse>(OpCode::kPullTensor, encode(r),
                                        decode_PullTensorResponse);
    }
};

struct DaemonFixture {
    std::string sock_path;
    std::unique_ptr<OffloadDaemon> daemon;
    std::thread thr;

    explicit DaemonFixture(uint64_t arena_bytes, uint64_t gran,
                           uint64_t pinned_quota = 0) {
        sock_path = "/tmp/ofld_canon_" + std::to_string(getpid()) + ".sock";
        DaemonConfig cfg = default_smoke_config(arena_bytes, 0, 0);
        cfg.socket_path = sock_path;
        cfg.allocation_granularity_bytes = gran;
        cfg.nvme_enabled = false;
        cfg.drain_on_d2h_complete = false;   // keep canonical bytes pinned
        cfg.v2_quota_max_pinned_bytes = pinned_quota;
        daemon.reset(new OffloadDaemon(cfg));
        thr = std::thread([this] { daemon->run(); });
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

// Build a canonical key. dp_rank is intentionally absent from the wire key.
CanonicalTensorKeyWire mk_key(const JobKeyWire& job, uint64_t version,
                              uint64_t param_id, uint32_t tp, int expert,
                              uint64_t nbytes) {
    CanonicalTensorKeyWire k;
    k.job = job;
    k.model_role = MODEL_ROLE_POLICY_ROLLOUT;
    k.model_version = version;
    k.param_id = param_id;
    k.param_fqn_hash = param_id * 2654435761u;
    k.param_fqn_debug = "layers.p" + std::to_string(param_id);
    k.dtype = 5; k.layout = 1; k.nbytes = nbytes;
    k.shape_hash = 0x1234; k.stride_hash = 0x5678;
    k.pp_rank = 0; k.tp_rank = tp; k.ep_rank = 0; k.etp_rank = 0;
    k.expert_id = expert;
    k.shard_offset = 0; k.shard_nbytes = nbytes;
    return k;
}

RequestCanonicalEvictRequest mk_evict(uint32_t rank, uint64_t epoch,
                                      const CanonicalTensorKeyWire& key,
                                      uint32_t dedup_mode, bool allow_dup = false) {
    RequestCanonicalEvictRequest r;
    r.rank_id = rank; r.rank_epoch = epoch; r.key = key;
    r.local_tensor_id = 1; r.local_version = 1;
    r.destructive = true; r.attach_if_exists = true; r.create_if_missing = true;
    r.dedup_mode = dedup_mode; r.allow_duplicate_candidate = allow_dup;
    r.alignment = 256; r.gpu_id = 0; r.numa_node = 0;
    return r;
}

// Drive create: NEED_D2H_CREATE -> D2H submitted/complete -> commit.
void do_create(RawClient& c, uint32_t rank, uint64_t epoch,
               const RequestCanonicalEvictResponse& ev, uint64_t hlo = 0,
               uint64_t hhi = 0) {
    uint64_t syn_tid = (1ull << 63) | ev.object_id;
    auto s = c.d2h_sub(rank, epoch, ev.lease_id, syn_tid, 1, ev.slot_id);
    CHECK(s.ok);
    auto d = c.d2h_done(rank, epoch, ev.lease_id, syn_tid, 1, ev.slot_id);
    CHECK(d.ok);
    auto cm = c.commit(rank, epoch, ev.object_id, ev.lease_id, ev.slot_id, hlo, hhi);
    CHECK(cm.ok);
    CHECK(cm.won);
}

// --------------------------------------------------------------------------
// §1 Job namespace: same name + different scheduler id => distinct job_id;
// two registrations of the same name get distinct launch_epochs.
// --------------------------------------------------------------------------
static void test_job_namespace(DaemonFixture& fx) {
    RawClient c(fx.sock_path);
    auto a = c.register_job("qwen_grpo", 1001);
    auto b = c.register_job("qwen_grpo", 2002);
    CHECK(a.ok); CHECK(b.ok);
    CHECK(a.job.job_id != b.job.job_id);         // different scheduler ids
    auto a2 = c.register_job("qwen_grpo", 1001);  // relaunch: same id, new epoch
    CHECK_EQ(a2.job.job_id, a.job.job_id);
    CHECK(a2.job.launch_epoch != a.job.launch_epoch);
}

// --------------------------------------------------------------------------
// §2 DP-equivalent attach: rank0 creates, rank1 (same key) ATTACHED_EXISTING,
//    no D2H. TP-distinct and expert-distinct keys do NOT attach.
// --------------------------------------------------------------------------
static void test_dp_attach_and_distinct(DaemonFixture& fx) {
    std::vector<int> fds0, fds1;
    RawClient c(fx.sock_path);
    auto j = c.register_job("dedup_job", 42).job;
    RawClient c0(fx.sock_path); auto r0 = c0.register_rank(0, &fds0);
    for (int fd : fds0) close_fd(fd);
    RawClient c1(fx.sock_path); auto r1 = c1.register_rank(1, &fds1);
    for (int fd : fds1) close_fd(fd);

    const uint64_t NB = 1u << 20;
    auto key = mk_key(j, /*version=*/7, /*param=*/100, /*tp=*/0, /*expert=*/-1, NB);

    // rank0: create
    auto ev0 = c0.can_evict(mk_evict(0, r0.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK(ev0.ok);
    CHECK_EQ(ev0.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);
    do_create(c0, 0, r0.rank_epoch, ev0);

    // rank1: same key (differs only in dp) -> attach, no d2h
    auto ev1 = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK(ev1.ok);
    CHECK_EQ(ev1.action, (uint32_t)ATTACH_ACTION_ATTACHED_EXISTING);
    CHECK_EQ(ev1.object_id, ev0.object_id);

    // TP-distinct -> separate object
    auto keytp = mk_key(j, 7, 100, /*tp=*/1, -1, NB);
    auto evtp = c0.can_evict(mk_evict(0, r0.rank_epoch, keytp, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(evtp.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);
    CHECK(evtp.object_id != ev0.object_id);
    do_create(c0, 0, r0.rank_epoch, evtp);

    // expert-distinct -> separate object
    auto keyx = mk_key(j, 7, 100, /*tp=*/0, /*expert=*/3, NB);
    auto evx = c0.can_evict(mk_evict(0, r0.rank_epoch, keyx, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(evx.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);
    CHECK(evx.object_id != ev0.object_id);
}

// --------------------------------------------------------------------------
// §3 Creating race: rank0 creates (not yet committed), rank1 with default
//    policy gets WAIT_FOR_CREATOR; after commit rank1 ATTACHED_EXISTING.
// --------------------------------------------------------------------------
static void test_creating_race(DaemonFixture& fx) {
    std::vector<int> f0, f1;
    RawClient c0(fx.sock_path); auto r0 = c0.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    RawClient c1(fx.sock_path); auto r1 = c1.register_rank(1, &f1);
    for (int fd : f1) close_fd(fd);
    auto j = c0.register_job("race_job", 77).job;
    const uint64_t NB = 1u << 20;
    auto key = mk_key(j, 1, 200, 0, -1, NB);

    auto ev0 = c0.can_evict(mk_evict(0, r0.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(ev0.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);
    // rank1 arrives while object is CREATING
    auto ev1 = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(ev1.action, (uint32_t)ATTACH_ACTION_WAIT_FOR_CREATOR);
    // creator finishes
    do_create(c0, 0, r0.rank_epoch, ev0);
    // rank1 retries -> now attaches
    auto ev1b = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(ev1b.action, (uint32_t)ATTACH_ACTION_ATTACHED_EXISTING);
    CHECK_EQ(ev1b.object_id, ev0.object_id);
}

// --------------------------------------------------------------------------
// §4 Seal + rollout pull to a file; §4 rollout-cannot-pull-unsealed.
// --------------------------------------------------------------------------
static void test_seal_and_pull(DaemonFixture& fx) {
    std::vector<int> f0;
    RawClient c(fx.sock_path); auto r0 = c.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    auto j = c.register_job("seal_job", 88).job;
    const uint64_t NB = 4096;
    auto k1 = mk_key(j, /*version=*/10, /*param=*/1, 0, -1, NB);
    auto k2 = mk_key(j, 10, 2, 0, -1, NB);

    auto e1 = c.can_evict(mk_evict(0, r0.rank_epoch, k1, DEDUP_SEMANTIC_TRUSTED));
    do_create(c, 0, r0.rank_epoch, e1, 0xAAAA, 0xBBBB);
    auto e2 = c.can_evict(mk_evict(0, r0.rank_epoch, k2, DEDUP_SEMANTIC_TRUSTED));
    do_create(c, 0, r0.rank_epoch, e2, 0xCCCC, 0xDDDD);

    // Cannot pull before seal.
    auto badpull = c.pull(j, MODEL_ROLE_POLICY_ROLLOUT, 10, e1.object_id, 2, "/tmp/x");
    CHECK(!badpull.ok);

    // Seal + promote.
    auto s = c.seal(j, MODEL_ROLE_POLICY_ROLLOUT, 10, /*promote=*/true,
                    /*fail_if_missing=*/true);
    CHECK(s.ok);
    CHECK_EQ(s.tensor_count, 2u);

    auto lv = c.latest(j, MODEL_ROLE_POLICY_ROLLOUT);
    CHECK(lv.found);
    CHECK_EQ(lv.model_version, 10u);

    auto man = c.manifest(j, MODEL_ROLE_POLICY_ROLLOUT, 10);
    CHECK(man.ok);
    CHECK_EQ(man.tensors.size(), 2u);
    CHECK_EQ(man.state, (uint32_t)MANIFEST_SEALED);

    // Pull object 1 to a file and verify header + payload size.
    std::string path = "/tmp/ofld_pull_" + std::to_string(getpid()) + ".bin";
    auto p = c.pull(j, MODEL_ROLE_POLICY_ROLLOUT, 10, e1.object_id, /*FILE=*/2, path);
    CHECK(p.ok);
    CHECK_EQ(p.nbytes, NB);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    CHECK(in.good());
    // header is 40 bytes (magic,ver,object_id,nbytes,hlo,hhi) + NB payload.
    CHECK_EQ((uint64_t)in.tellg(), (uint64_t)(40 + NB));
    in.close();
    ::unlink(path.c_str());
}

// --------------------------------------------------------------------------
// §4 Seal fails when a version has no valid objects and fail_if_missing set
//    with expected_param_ids missing. Here: sealing an empty version.
// --------------------------------------------------------------------------
static void test_seal_empty_ok_but_pull_rejects_unsealed(DaemonFixture& fx) {
    RawClient c(fx.sock_path);
    auto j = c.register_job("empty_seal", 89).job;
    // Sealing a version with zero objects succeeds (nothing missing) but has 0
    // tensors; a rollout pull of a non-existent object must still be rejected.
    auto s = c.seal(j, MODEL_ROLE_POLICY_ROLLOUT, 99, true, true);
    CHECK(s.ok);
    CHECK_EQ(s.tensor_count, 0u);
    auto p = c.pull(j, MODEL_ROLE_POLICY_ROLLOUT, 99, 123456, 2, "/tmp/none");
    CHECK(!p.ok);  // unknown object
}

// --------------------------------------------------------------------------
// §6 Per-job pinned quota: a tiny quota rejects the create with QUOTA_EXCEEDED.
// --------------------------------------------------------------------------
static void test_quota(DaemonFixture& fx) {
    std::vector<int> f0;
    RawClient c(fx.sock_path); auto r0 = c.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    auto j = c.register_job("quota_job", 90).job;
    // quota is set on the fixture; request a tensor bigger than it.
    const uint64_t NB = 8u << 20;   // 8 MiB, quota is 1 MiB
    auto key = mk_key(j, 1, 500, 0, -1, NB);
    auto ev = c.can_evict(mk_evict(0, r0.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK(!ev.ok);
    CHECK_EQ(ev.action, (uint32_t)ATTACH_ACTION_QUOTA_EXCEEDED);
}

// --------------------------------------------------------------------------
// §5 Export over the TCP debug transport: stand up a tiny TCP receiver, seal,
// then PullTensor(TCP) and verify the received header + payload bytes.
// --------------------------------------------------------------------------
static void test_pull_over_tcp(DaemonFixture& fx) {
    std::vector<int> f0;
    RawClient c(fx.sock_path); auto r0 = c.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    auto j = c.register_job("tcp_pull_job", 91).job;
    const uint64_t NB = 4096;
    auto k = mk_key(j, /*version=*/5, /*param=*/1, 0, -1, NB);
    auto e = c.can_evict(mk_evict(0, r0.rank_epoch, k, DEDUP_SEMANTIC_TRUSTED));
    do_create(c, 0, r0.rank_epoch, e, 0x1234, 0x5678);
    auto s = c.seal(j, MODEL_ROLE_POLICY_ROLLOUT, 5, true, true);
    CHECK(s.ok);

    // Bind an ephemeral TCP port; hand its address to PullTensor.
    int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(lfd >= 0);
    int one = 1; ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr; std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    CHECK(::bind(lfd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    CHECK(::listen(lfd, 4) == 0);
    socklen_t alen = sizeof(addr);
    ::getsockname(lfd, (struct sockaddr*)&addr, &alen);
    uint16_t port = ntohs(addr.sin_port);

    // Receiver thread: accept one connection, read 40-byte header + NB payload.
    uint64_t got_bytes = 0, got_object = 0;
    bool magic_ok = false;
    std::thread rx([&]{
        int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) return;
        uint8_t hdr[40];
        size_t off = 0;
        while (off < sizeof(hdr)) {
            ssize_t n = ::read(cfd, hdr + off, sizeof(hdr) - off);
            if (n <= 0) break;
            off += n;
        }
        uint32_t magic; std::memcpy(&magic, hdr, 4);
        magic_ok = (magic == 0x4F464558u);  // 'OFEX'
        std::memcpy(&got_object, hdr + 8, 8);
        std::memcpy(&got_bytes, hdr + 16, 8);
        // drain payload
        std::vector<uint8_t> buf(NB); off = 0;
        while (off < NB) {
            ssize_t n = ::read(cfd, buf.data() + off, NB - off);
            if (n <= 0) break;
            off += n;
        }
        ::close(cfd);
    });

    std::string dst = "127.0.0.1:" + std::to_string(port);
    auto p = c.pull(j, MODEL_ROLE_POLICY_ROLLOUT, 5, e.object_id, /*TCP=*/1, dst);
    rx.join();
    ::close(lfd);
    CHECK(p.ok);
    CHECK(magic_ok);
    CHECK_EQ(got_bytes, NB);
    CHECK_EQ(got_object, e.object_id);
}

// --------------------------------------------------------------------------
// §2 Hash-verified dedup: a corrupt replica (different content hash) is
// rejected on commit; a matching replica attaches.
// --------------------------------------------------------------------------
static void test_hash_verified_mismatch(DaemonFixture& fx) {
    std::vector<int> f0, f1;
    RawClient c0(fx.sock_path); auto r0 = c0.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    RawClient c1(fx.sock_path); auto r1 = c1.register_rank(1, &f1);
    for (int fd : f1) close_fd(fd);
    auto j = c0.register_job("hashv_job", 61).job;
    const uint64_t NB = 1u << 20;
    auto key = mk_key(j, 1, 300, 0, -1, NB);

    // rank0 creates with content hash (0xAA, 0xBB).
    auto ev0 = c0.can_evict(mk_evict(0, r0.rank_epoch, key, DEDUP_HASH_VERIFIED));
    CHECK_EQ(ev0.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);
    do_create(c0, 0, r0.rank_epoch, ev0, 0xAA, 0xBB);

    // rank1 (hash-verified) gets a verify candidate slot, D2H, then commits.
    auto ev1 = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_HASH_VERIFIED));
    CHECK_EQ(ev1.action, (uint32_t)ATTACH_ACTION_DUPLICATE_CANDIDATE);
    CHECK_EQ(ev1.object_id, ev0.object_id);
    uint64_t cand_tid = ev1.synthetic_tid;
    CHECK(c1.d2h_sub(1, r1.rank_epoch, ev1.lease_id, cand_tid, 1, ev1.slot_id).ok);
    CHECK(c1.d2h_done(1, r1.rank_epoch, ev1.lease_id, cand_tid, 1, ev1.slot_id).ok);
    // Commit with a MISMATCHING hash -> rejected.
    auto bad = c1.commit(1, r1.rank_epoch, ev1.object_id, ev1.lease_id,
                         ev1.slot_id, 0xDE, 0xAD);
    CHECK(!bad.ok);
    CHECK(!bad.won);

    // A second verify candidate committing the MATCHING hash attaches.
    auto ev2 = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_HASH_VERIFIED));
    CHECK_EQ(ev2.action, (uint32_t)ATTACH_ACTION_DUPLICATE_CANDIDATE);
    uint64_t cand2 = ev2.synthetic_tid;
    CHECK(c1.d2h_sub(1, r1.rank_epoch, ev2.lease_id, cand2, 1, ev2.slot_id).ok);
    CHECK(c1.d2h_done(1, r1.rank_epoch, ev2.lease_id, cand2, 1, ev2.slot_id).ok);
    auto good = c1.commit(1, r1.rank_epoch, ev2.object_id, ev2.lease_id,
                          ev2.slot_id, 0xAA, 0xBB);
    CHECK(good.ok);
    CHECK(!good.won);   // attached to the winner, did not become the object
}

// --------------------------------------------------------------------------
// §3 Duplicate candidate under pressure: with allow_duplicate_candidate, a
// second rank racing a CREATING object gets its own candidate slot; the first
// commit wins, the loser attaches (won=false) and its candidate is reclaimed.
// --------------------------------------------------------------------------
static void test_duplicate_candidate(DaemonFixture& fx) {
    std::vector<int> f0, f1;
    RawClient c0(fx.sock_path); auto r0 = c0.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    RawClient c1(fx.sock_path); auto r1 = c1.register_rank(1, &f1);
    for (int fd : f1) close_fd(fd);
    auto j = c0.register_job("dupcand_job", 62).job;
    const uint64_t NB = 1u << 20;
    auto key = mk_key(j, 1, 400, 0, -1, NB);

    // rank0 creates but does NOT commit yet (object is CREATING).
    auto ev0 = c0.can_evict(mk_evict(0, r0.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED));
    CHECK_EQ(ev0.action, (uint32_t)ATTACH_ACTION_NEED_D2H_CREATE);

    // rank1 with allow_duplicate_candidate -> its own candidate slot.
    auto ev1 = c1.can_evict(mk_evict(1, r1.rank_epoch, key, DEDUP_SEMANTIC_TRUSTED,
                                     /*allow_dup=*/true));
    CHECK_EQ(ev1.action, (uint32_t)ATTACH_ACTION_DUPLICATE_CANDIDATE);
    CHECK(ev1.synthetic_tid != ev0.synthetic_tid);   // distinct backing slots

    // rank0 wins: finish D2H + commit the real object.
    do_create(c0, 0, r0.rank_epoch, ev0);

    // rank1 finishes its candidate D2H then commits -> attaches (loser).
    CHECK(c1.d2h_sub(1, r1.rank_epoch, ev1.lease_id, ev1.synthetic_tid, 1, ev1.slot_id).ok);
    CHECK(c1.d2h_done(1, r1.rank_epoch, ev1.lease_id, ev1.synthetic_tid, 1, ev1.slot_id).ok);
    auto loser = c1.commit(1, r1.rank_epoch, ev1.object_id, ev1.lease_id, ev1.slot_id);
    CHECK(loser.ok);
    CHECK(!loser.won);
}

// --------------------------------------------------------------------------
// §4 Seal fails when expected_param_ids are not all present + fail_if_missing.
// --------------------------------------------------------------------------
static void test_seal_fail_on_missing(DaemonFixture& fx) {
    std::vector<int> f0;
    RawClient c(fx.sock_path); auto r0 = c.register_rank(0, &f0);
    for (int fd : f0) close_fd(fd);
    auto j = c.register_job("sealmiss_job", 63).job;
    const uint64_t NB = 4096;
    // Create only param 1 of version 55.
    auto k1 = mk_key(j, 55, 1, 0, -1, NB);
    auto e1 = c.can_evict(mk_evict(0, r0.rank_epoch, k1, DEDUP_SEMANTIC_TRUSTED));
    do_create(c, 0, r0.rank_epoch, e1);

    // Seal requiring params {1,2} must fail (2 missing).
    SealModelVersionRequest sr;
    sr.job = j; sr.model_role = MODEL_ROLE_POLICY_ROLLOUT; sr.model_version = 55;
    sr.fail_if_missing = true; sr.expected_param_ids = {1, 2}; sr.promote = false;
    auto bad = c.call<SealModelVersionResponse>(OpCode::kSealModelVersion,
        encode(sr), decode_SealModelVersionResponse);
    CHECK(!bad.ok);
    CHECK_EQ(bad.state, (uint32_t)MANIFEST_ERROR);

    // A rollout pull of this (unsealed) version is refused.
    auto p = c.pull(j, MODEL_ROLE_POLICY_ROLLOUT, 55, e1.object_id, 2, "/tmp/x");
    CHECK(!p.ok);
}

}  // namespace

int main() {
    {
        DaemonFixture fx(256u << 20, 1u << 20);
        RUN([&]{ test_job_namespace(fx); });
        RUN([&]{ test_dp_attach_and_distinct(fx); });
        RUN([&]{ test_creating_race(fx); });
        RUN([&]{ test_hash_verified_mismatch(fx); });
        RUN([&]{ test_duplicate_candidate(fx); });
        RUN([&]{ test_seal_and_pull(fx); });
        RUN([&]{ test_seal_fail_on_missing(fx); });
        RUN([&]{ test_pull_over_tcp(fx); });
        RUN([&]{ test_seal_empty_ok_but_pull_rejects_unsealed(fx); });
    }
    {
        // Separate fixture with a 1 MiB per-job pinned quota.
        DaemonFixture fx(256u << 20, 1u << 20, /*pinned_quota=*/1u << 20);
        RUN([&]{ test_quota(fx); });
    }
    return ofldtest::summary("canonical");
}
