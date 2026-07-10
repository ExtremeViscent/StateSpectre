// Unit test: wire codec round-trips + framing (TEST_PLAN §1).
#include "test_harness.h"
#include "wire.h"

using namespace offload;
using namespace ofldtest;

// Encode -> frame -> parse header -> decode, and assert equality of fields.
static void test_register_roundtrip() {
    RegisterRankRequest req;
    req.rank_id = 7; req.local_rank = 1; req.world_rank = 15;
    req.gpu_id = 3; req.numa_node = 1; req.pid = 123456; req.capabilities = 0xABCD;
    auto body = encode(req);
    auto frame = make_frame(OpCode::kRegisterRank, body);
    auto h = parse_frame_header(frame.data(), frame.size());
    CHECK_EQ(h.magic, OFLD_RPC_MAGIC);
    CHECK_EQ(h.opcode, (uint16_t)OpCode::kRegisterRank);
    CHECK_EQ(h.payload_len, (uint32_t)body.size());
    auto got = decode_RegisterRankRequest(frame.data() + kFrameHeaderSize, h.payload_len);
    CHECK_EQ(got.rank_id, 7u);
    CHECK_EQ(got.world_rank, 15u);
    CHECK_EQ(got.gpu_id, 3u);
    CHECK_EQ(got.numa_node, 1u);
    CHECK_EQ(got.pid, 123456ull);
    CHECK_EQ(got.capabilities, 0xABCDull);
}

static void test_register_response_arenas() {
    RegisterRankResponse resp;
    resp.ok = true; resp.message = "welcome";
    resp.rank_epoch = 42; resp.control_generation = 9;
    resp.control_fd_index = 2; resp.control_size = (1<<20); resp.num_slots = 128;
    ArenaFd a0; a0.arena_id = 100; a0.numa_node = 0; a0.kind = 1;
    a0.size = (64ull<<20); a0.base_offset = 0; a0.preferred_gpu = 0;
    a0.flags = 5; a0.fd_index = 0; a0.registration_granularity = (512u<<20);
    a0.allocation_granularity = (2u<<20);
    ArenaFd a1 = a0; a1.arena_id = 101; a1.preferred_gpu = 1; a1.fd_index = 1;
    resp.arenas = {a0, a1};

    auto b = encode(resp);
    auto got = decode_RegisterRankResponse(b.data(), b.size());
    CHECK(got.ok);
    CHECK_STREQ(got.message, "welcome");
    CHECK_EQ(got.rank_epoch, 42ull);
    CHECK_EQ(got.num_slots, 128u);
    CHECK_EQ(got.control_size, (uint64_t)(1<<20));
    CHECK_EQ((int)got.arenas.size(), 2);
    CHECK_EQ(got.arenas[0].arena_id, 100ull);
    CHECK_EQ(got.arenas[1].arena_id, 101ull);
    CHECK_EQ(got.arenas[1].preferred_gpu, 1u);
    CHECK_EQ(got.arenas[0].allocation_granularity, (uint32_t)(2u<<20));
}

static void test_offload_roundtrip() {
    RequestOffloadRequest req;
    req.rank_id = 2; req.rank_epoch = 5; req.tensor_id = 0xDEADBEEF;
    req.version = 11; req.nbytes = (80ull<<30); req.gpu_id = 4; req.numa_node = 1;
    req.alignment = 256; req.priority = 1; req.flags = 0x18;
    req.debug_name = "k_cache";
    auto b = encode(req);
    auto got = decode_RequestOffloadRequest(b.data(), b.size());
    CHECK_EQ(got.tensor_id, 0xDEADBEEFull);
    CHECK_EQ(got.version, 11ull);
    CHECK_EQ(got.nbytes, (80ull<<30));
    CHECK_EQ(got.flags, 0x18ull);
    CHECK_STREQ(got.debug_name, "k_cache");

    RequestOffloadResponse resp;
    resp.ok = true; resp.message = ""; resp.lease_id = 999; resp.slot_id = 77;
    resp.arena_id = 101; resp.arena_offset = (16ull<<20); resp.capacity = (32ull<<20);
    resp.state = OFLD_SLOT_RESERVED_D2H; resp.blocked = false;
    auto rb = encode(resp);
    auto rg = decode_RequestOffloadResponse(rb.data(), rb.size());
    CHECK(rg.ok);
    CHECK_EQ(rg.lease_id, 999ull);
    CHECK_EQ(rg.slot_id, 77u);
    CHECK_EQ(rg.arena_offset, (uint64_t)(16ull<<20));
    CHECK_EQ(rg.state, (uint32_t)OFLD_SLOT_RESERVED_D2H);
    CHECK(!rg.blocked);
}

static void test_d2h_complete_flags() {
    MarkD2HCompleteResponse r;
    r.ok = true; r.message = "ok"; r.latest_version = true; r.drain_enqueued = false;
    auto b = encode(r);
    auto g = decode_MarkD2HCompleteResponse(b.data(), b.size());
    CHECK(g.ok);
    CHECK(g.latest_version);
    CHECK(!g.drain_enqueued);
}

static void test_batch_roundtrip() {
    BatchCompleteRequest req;
    req.rank_id = 1; req.rank_epoch = 3;
    for (int i = 0; i < 5; ++i) {
        BatchCompletion c;
        c.event_type = (uint32_t)EventType::kD2HComplete;
        c.lease_id = 1000 + i; c.tensor_id = i; c.version = 1;
        c.slot_id = i; c.seq = i * 10; c.timestamp_ns = 555 + i;
        c.keep_pinned = (i % 2 == 0);
        req.completions.push_back(c);
    }
    auto b = encode(req);
    auto g = decode_BatchCompleteRequest(b.data(), b.size());
    CHECK_EQ((int)g.completions.size(), 5);
    CHECK_EQ(g.completions[3].lease_id, 1003ull);
    CHECK_EQ(g.completions[4].seq, 40ull);
    CHECK_EQ(g.completions[2].keep_pinned, true);
    CHECK_EQ(g.completions[3].keep_pinned, false);
}

static void test_stats_roundtrip() {
    GetStatsResponse r;
    r.ok = true; r.message = "ok";
    r.keys = {"d2h_bytes", "used_pinned_bytes"};
    r.values = {123456789ull, 42ull};
    auto b = encode(r);
    auto g = decode_GetStatsResponse(b.data(), b.size());
    CHECK_EQ((int)g.keys.size(), 2);
    CHECK_STREQ(g.keys[0], "d2h_bytes");
    CHECK_EQ(g.values[0], 123456789ull);
    CHECK_EQ(g.values[1], 42ull);
}

static void test_bad_magic_rejected() {
    std::vector<uint8_t> junk(kFrameHeaderSize, 0);
    bool threw = false;
    try {
        parse_frame_header(junk.data(), junk.size());
    } catch (const WireError&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_truncation_rejected() {
    RequestOffloadRequest req;
    req.debug_name = "abc";
    auto b = encode(req);
    bool threw = false;
    try {
        // Feed 3 fewer bytes than encoded => reader must throw.
        decode_RequestOffloadRequest(b.data(), b.size() - 3);
    } catch (const WireError&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    RUN(test_register_roundtrip);
    RUN(test_register_response_arenas);
    RUN(test_offload_roundtrip);
    RUN(test_d2h_complete_flags);
    RUN(test_batch_roundtrip);
    RUN(test_stats_roundtrip);
    RUN(test_bad_magic_rejected);
    RUN(test_truncation_rejected);
    return summary("wire");
}
