// Round-trip codec tests for the v2 canonical model-state wire messages.
#include "test_harness.h"

#include "protocol_v2.h"
#include "wire_v2.h"

using namespace offload;

static JobKeyWire mk_job() {
    JobKeyWire j;
    j.tenant_id = 7; j.job_id = 912884112ull; j.launch_epoch = 3;
    j.job_name = "qwen32b_grpo_math";
    return j;
}

static CanonicalTensorKeyWire mk_key() {
    CanonicalTensorKeyWire k;
    k.job = mk_job();
    k.model_role = 2; k.model_version = 1042;
    k.param_id = 3012; k.param_fqn_hash = 0xdeadbeefcafef00dull;
    k.param_fqn_debug = "layers.12.mlp.w1.weight";
    k.dtype = 5; k.layout = 1; k.nbytes = 117440512ull;
    k.shape_hash = 0x1111; k.stride_hash = 0x2222;
    k.pp_rank = 1; k.tp_rank = 3; k.ep_rank = 0; k.etp_rank = 0; k.expert_id = -1;
    k.shard_offset = 0; k.shard_nbytes = 117440512ull;
    return k;
}

static void test_register_job() {
    RegisterJobRequest req;
    req.tenant_id = 7; req.job_name = "job-a"; req.scheduler_job_id = 555;
    req.label_keys = {"k1", "k2"}; req.label_values = {"v1", "v2"};
    auto b = encode(req);
    auto d = decode_RegisterJobRequest(b.data(), b.size());
    CHECK_EQ(d.tenant_id, 7u);
    CHECK_STREQ(d.job_name, "job-a");
    CHECK_EQ(d.scheduler_job_id, 555u);
    CHECK_EQ(d.label_keys.size(), 2u);
    CHECK_STREQ(d.label_values[1], "v2");

    RegisterJobResponse resp;
    resp.ok = true; resp.message = "ok"; resp.job = mk_job(); resp.control_generation = 99;
    auto rb = encode(resp);
    auto rd = decode_RegisterJobResponse(rb.data(), rb.size());
    CHECK(rd.ok);
    CHECK_EQ(rd.job.job_id, 912884112ull);
    CHECK_STREQ(rd.job.job_name, "qwen32b_grpo_math");
    CHECK_EQ(rd.control_generation, 99u);
}

static void test_canonical_evict() {
    RequestCanonicalEvictRequest req;
    req.rank_id = 5; req.rank_epoch = 42; req.key = mk_key();
    req.local_tensor_id = 100; req.local_version = 7;
    req.destructive = true; req.attach_if_exists = true; req.create_if_missing = true;
    req.dedup_mode = 1; req.allow_duplicate_candidate = true;
    req.alignment = 256; req.priority = 2; req.gpu_id = 1; req.numa_node = 0;
    auto b = encode(req);
    auto d = decode_RequestCanonicalEvictRequest(b.data(), b.size());
    CHECK_EQ(d.rank_id, 5u);
    CHECK_EQ(d.rank_epoch, 42u);
    CHECK_EQ(d.key.tp_rank, 3u);
    CHECK_EQ(d.key.expert_id, -1);
    CHECK_EQ(d.key.nbytes, 117440512ull);
    CHECK(d.allow_duplicate_candidate);
    CHECK_EQ(d.gpu_id, 1u);
    CHECK_STREQ(d.key.param_fqn_debug, "layers.12.mlp.w1.weight");

    RequestCanonicalEvictResponse resp;
    resp.ok = true; resp.message = "ok"; resp.action = 1;
    resp.object_id = 998812; resp.lease_id = 33; resp.slot_id = 12;
    resp.arena_id = 0; resp.arena_offset = 4096; resp.capacity = 1u << 20;
    auto rb = encode(resp);
    auto rd = decode_RequestCanonicalEvictResponse(rb.data(), rb.size());
    CHECK_EQ(rd.action, 1u);
    CHECK_EQ(rd.object_id, 998812u);
    CHECK_EQ(rd.capacity, (uint64_t)(1u << 20));
}

static void test_commit() {
    CommitCanonicalObjectRequest req;
    req.rank_id = 5; req.rank_epoch = 42; req.object_id = 998812; req.lease_id = 33;
    req.slot_id = 12; req.content_hash_lo = 0xaaaa; req.content_hash_hi = 0xbbbb;
    auto b = encode(req);
    auto d = decode_CommitCanonicalObjectRequest(b.data(), b.size());
    CHECK_EQ(d.object_id, 998812u);
    CHECK_EQ(d.content_hash_hi, 0xbbbbu);

    CommitCanonicalObjectResponse resp;
    resp.ok = true; resp.message = "ok"; resp.object_id = 998812; resp.won = false;
    auto rb = encode(resp);
    auto rd = decode_CommitCanonicalObjectResponse(rb.data(), rb.size());
    CHECK(rd.ok);
    CHECK(!rd.won);
}

static void test_seal_and_latest() {
    SealModelVersionRequest req;
    req.job = mk_job(); req.model_role = 2; req.model_version = 1042;
    req.fail_if_missing = true; req.expected_param_ids = {1, 2, 3012}; req.promote = true;
    auto b = encode(req);
    auto d = decode_SealModelVersionRequest(b.data(), b.size());
    CHECK_EQ(d.model_version, 1042u);
    CHECK_EQ(d.expected_param_ids.size(), 3u);
    CHECK_EQ(d.expected_param_ids[2], 3012u);
    CHECK(d.promote);

    SealModelVersionResponse resp;
    resp.ok = true; resp.state = 2; resp.tensor_count = 100; resp.total_nbytes = 1ull << 34;
    resp.manifest_hash_lo = 1; resp.manifest_hash_hi = 2;
    auto rb = encode(resp);
    auto rd = decode_SealModelVersionResponse(rb.data(), rb.size());
    CHECK_EQ(rd.state, 2u);
    CHECK_EQ(rd.total_nbytes, (uint64_t)(1ull << 34));

    GetLatestSealedVersionRequest lreq; lreq.job = mk_job(); lreq.model_role = 2;
    auto lb = encode(lreq);
    auto ld = decode_GetLatestSealedVersionRequest(lb.data(), lb.size());
    CHECK_EQ(ld.model_role, 2u);
    GetLatestSealedVersionResponse lresp;
    lresp.ok = true; lresp.found = true; lresp.model_version = 1042;
    auto lrb = encode(lresp);
    auto lrd = decode_GetLatestSealedVersionResponse(lrb.data(), lrb.size());
    CHECK(lrd.found);
    CHECK_EQ(lrd.model_version, 1042u);
}

static void test_manifest() {
    GetManifestRequest req;
    req.job = mk_job(); req.model_role = 2; req.model_version = 1042;
    req.filter.tp_present = true; req.filter.tp_rank = 3;
    req.filter.expert_present = true; req.filter.expert_id = -1;
    auto b = encode(req);
    auto d = decode_GetManifestRequest(b.data(), b.size());
    CHECK(d.filter.tp_present);
    CHECK_EQ(d.filter.tp_rank, 3u);
    CHECK(d.filter.expert_present);
    CHECK_EQ(d.filter.expert_id, -1);
    CHECK(!d.filter.pp_present);

    GetManifestResponse resp;
    resp.ok = true; resp.job = mk_job(); resp.model_role = 2; resp.model_version = 1042;
    resp.state = 2;
    TensorManifestEntryWire e;
    e.key = mk_key(); e.object_id = 998812; e.nbytes = 117440512ull;
    e.content_hash_lo = 7; e.content_hash_hi = 8; e.location_hint = "node3:nvme";
    resp.tensors.push_back(e);
    resp.tensors.push_back(e);
    auto rb = encode(resp);
    auto rd = decode_GetManifestResponse(rb.data(), rb.size());
    CHECK_EQ(rd.tensors.size(), 2u);
    CHECK_EQ(rd.tensors[0].object_id, 998812u);
    CHECK_STREQ(rd.tensors[1].location_hint, "node3:nvme");
    CHECK_EQ(rd.tensors[0].key.tp_rank, 3u);
}

static void test_pull() {
    PullTensorRequest req;
    req.job = mk_job(); req.model_role = 2; req.model_version = 1042;
    req.object_id = 998812; req.transport = 3; req.target_descriptor = "host:19090";
    auto b = encode(req);
    auto d = decode_PullTensorRequest(b.data(), b.size());
    CHECK_EQ(d.object_id, 998812u);
    CHECK_EQ(d.transport, 3u);
    CHECK_STREQ(d.target_descriptor, "host:19090");

    PullTensorResponse resp;
    resp.ok = true; resp.object_id = 998812; resp.nbytes = 117440512ull;
    resp.transport = 3; resp.content_hash_lo = 7; resp.content_hash_hi = 8;
    resp.transport_metadata = "xfer-123";
    auto rb = encode(resp);
    auto rd = decode_PullTensorResponse(rb.data(), rb.size());
    CHECK(rd.ok);
    CHECK_EQ(rd.nbytes, 117440512ull);
    CHECK_STREQ(rd.transport_metadata, "xfer-123");
}

static void test_frame_roundtrip() {
    // The v2 messages ride the same frame envelope as v1.
    PullTensorRequest req;
    req.job = mk_job(); req.object_id = 42; req.transport = 1;
    auto body = encode(req);
    auto frame = make_frame(OpCode::kPullTensor, body);
    auto h = parse_frame_header(frame.data(), frame.size());
    CHECK_EQ(h.opcode, static_cast<uint16_t>(OpCode::kPullTensor));
    CHECK_EQ(h.payload_len, (uint32_t)body.size());
    auto d = decode_PullTensorRequest(frame.data() + kFrameHeaderSize, h.payload_len);
    CHECK_EQ(d.object_id, 42u);
}

static void test_canonical_restore() {
    RequestCanonicalRestoreRequest req;
    req.rank_id = 3; req.rank_epoch = 9; req.object_id = 998812;
    req.gpu_id = 1; req.numa_node = 0;
    auto b = encode(req);
    auto d = decode_RequestCanonicalRestoreRequest(b.data(), b.size());
    CHECK_EQ(d.rank_id, 3u);
    CHECK_EQ(d.object_id, 998812u);
    CHECK_EQ(d.gpu_id, 1u);

    RequestCanonicalRestoreResponse resp;
    resp.ok = true; resp.object_id = 998812; resp.nbytes = 117440512ull;
    resp.arena_id = 2; resp.arena_offset = 4096;
    auto rb = encode(resp);
    auto rd = decode_RequestCanonicalRestoreResponse(rb.data(), rb.size());
    CHECK(rd.ok);
    CHECK_EQ(rd.nbytes, 117440512ull);
    CHECK_EQ(rd.arena_id, 2u);
    CHECK_EQ(rd.arena_offset, 4096u);

    ReleaseCanonicalRestoreRequest rel;
    rel.rank_id = 3; rel.rank_epoch = 9; rel.object_id = 998812;
    auto lb = encode(rel);
    auto ld = decode_ReleaseCanonicalRestoreRequest(lb.data(), lb.size());
    CHECK_EQ(ld.object_id, 998812u);
    ReleaseCanonicalRestoreResponse lresp; lresp.ok = true; lresp.message = "ok";
    auto lrb = encode(lresp);
    auto lrd = decode_ReleaseCanonicalRestoreResponse(lrb.data(), lrb.size());
    CHECK(lrd.ok);
}

int main() {
    RUN(test_register_job);
    RUN(test_canonical_evict);
    RUN(test_commit);
    RUN(test_seal_and_latest);
    RUN(test_manifest);
    RUN(test_pull);
    RUN(test_frame_roundtrip);
    RUN(test_canonical_restore);
    return ofldtest::summary("wire_v2");
}
