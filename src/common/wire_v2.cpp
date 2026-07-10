// Codecs for the v2 canonical model-state messages. Mirrors protocol_v2.h and
// rpc/offload_canonical.proto. Reuses Writer/Reader from wire.h.

#include "wire_v2.h"

namespace offload {

namespace {

// ---- shared sub-message helpers -------------------------------------------
void put_jobkey(Writer& w, const JobKeyWire& j) {
    w.u64(j.tenant_id);
    w.u64(j.job_id);
    w.u64(j.launch_epoch);
    w.str(j.job_name);
}
JobKeyWire get_jobkey(Reader& r) {
    JobKeyWire j;
    j.tenant_id = r.u64();
    j.job_id = r.u64();
    j.launch_epoch = r.u64();
    j.job_name = r.str();
    return j;
}

void put_key(Writer& w, const CanonicalTensorKeyWire& k) {
    put_jobkey(w, k.job);
    w.u32(k.model_role);
    w.u64(k.model_version);
    w.u64(k.param_id);
    w.u64(k.param_fqn_hash);
    w.str(k.param_fqn_debug);
    w.u32(k.dtype);
    w.u32(k.layout);
    w.u64(k.nbytes);
    w.u64(k.shape_hash);
    w.u64(k.stride_hash);
    w.u32(k.pp_rank);
    w.u32(k.tp_rank);
    w.u32(k.ep_rank);
    w.u32(k.etp_rank);
    w.u32(static_cast<uint32_t>(k.expert_id));
    w.u64(k.shard_offset);
    w.u64(k.shard_nbytes);
}
CanonicalTensorKeyWire get_key(Reader& r) {
    CanonicalTensorKeyWire k;
    k.job = get_jobkey(r);
    k.model_role = r.u32();
    k.model_version = r.u64();
    k.param_id = r.u64();
    k.param_fqn_hash = r.u64();
    k.param_fqn_debug = r.str();
    k.dtype = r.u32();
    k.layout = r.u32();
    k.nbytes = r.u64();
    k.shape_hash = r.u64();
    k.stride_hash = r.u64();
    k.pp_rank = r.u32();
    k.tp_rank = r.u32();
    k.ep_rank = r.u32();
    k.etp_rank = r.u32();
    k.expert_id = static_cast<int32_t>(r.u32());
    k.shard_offset = r.u64();
    k.shard_nbytes = r.u64();
    return k;
}

}  // namespace

// ---- RegisterJob ----------------------------------------------------------
std::vector<uint8_t> encode(const RegisterJobRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u64(m.tenant_id);
    w.str(m.job_name);
    w.u64(m.scheduler_job_id);
    w.u32(static_cast<uint32_t>(m.label_keys.size()));
    for (const auto& s : m.label_keys) w.str(s);
    w.u32(static_cast<uint32_t>(m.label_values.size()));
    for (const auto& s : m.label_values) w.str(s);
    return b;
}
RegisterJobRequest decode_RegisterJobRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); RegisterJobRequest m;
    m.tenant_id = r.u64();
    m.job_name = r.str();
    m.scheduler_job_id = r.u64();
    uint32_t nk = r.u32();
    m.label_keys.reserve(nk);
    for (uint32_t i = 0; i < nk; ++i) m.label_keys.push_back(r.str());
    uint32_t nv = r.u32();
    m.label_values.reserve(nv);
    for (uint32_t i = 0; i < nv; ++i) m.label_values.push_back(r.str());
    return m;
}

std::vector<uint8_t> encode(const RegisterJobResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    put_jobkey(w, m.job);
    w.u64(m.control_generation);
    return b;
}
RegisterJobResponse decode_RegisterJobResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); RegisterJobResponse m;
    m.ok = r.b(); m.message = r.str();
    m.job = get_jobkey(r);
    m.control_generation = r.u64();
    return m;
}

// ---- RequestCanonicalEvict ------------------------------------------------
std::vector<uint8_t> encode(const RequestCanonicalEvictRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch);
    put_key(w, m.key);
    w.u64(m.local_tensor_id); w.u64(m.local_version);
    w.b(m.destructive); w.b(m.attach_if_exists); w.b(m.create_if_missing);
    w.u32(m.dedup_mode);
    w.b(m.allow_duplicate_candidate);
    w.u32(m.alignment); w.u32(m.priority); w.u32(m.gpu_id); w.u32(m.numa_node);
    return b;
}
RequestCanonicalEvictRequest decode_RequestCanonicalEvictRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestCanonicalEvictRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64();
    m.key = get_key(r);
    m.local_tensor_id = r.u64(); m.local_version = r.u64();
    m.destructive = r.b(); m.attach_if_exists = r.b(); m.create_if_missing = r.b();
    m.dedup_mode = r.u32();
    m.allow_duplicate_candidate = r.b();
    m.alignment = r.u32(); m.priority = r.u32(); m.gpu_id = r.u32(); m.numa_node = r.u32();
    return m;
}

std::vector<uint8_t> encode(const RequestCanonicalEvictResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u32(m.action);
    w.u64(m.object_id); w.u64(m.lease_id); w.u32(m.slot_id);
    w.u64(m.arena_id); w.u64(m.arena_offset); w.u64(m.capacity);
    return b;
}
RequestCanonicalEvictResponse decode_RequestCanonicalEvictResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestCanonicalEvictResponse m;
    m.ok = r.b(); m.message = r.str();
    m.action = r.u32();
    m.object_id = r.u64(); m.lease_id = r.u64(); m.slot_id = r.u32();
    m.arena_id = r.u64(); m.arena_offset = r.u64(); m.capacity = r.u64();
    return m;
}

// ---- CommitCanonicalObject ------------------------------------------------
std::vector<uint8_t> encode(const CommitCanonicalObjectRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch);
    w.u64(m.object_id); w.u64(m.lease_id); w.u32(m.slot_id);
    w.u64(m.content_hash_lo); w.u64(m.content_hash_hi);
    return b;
}
CommitCanonicalObjectRequest decode_CommitCanonicalObjectRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); CommitCanonicalObjectRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64();
    m.object_id = r.u64(); m.lease_id = r.u64(); m.slot_id = r.u32();
    m.content_hash_lo = r.u64(); m.content_hash_hi = r.u64();
    return m;
}

std::vector<uint8_t> encode(const CommitCanonicalObjectResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message); w.u64(m.object_id); w.b(m.won);
    return b;
}
CommitCanonicalObjectResponse decode_CommitCanonicalObjectResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); CommitCanonicalObjectResponse m;
    m.ok = r.b(); m.message = r.str(); m.object_id = r.u64(); m.won = r.b();
    return m;
}

// ---- SealModelVersion -----------------------------------------------------
std::vector<uint8_t> encode(const SealModelVersionRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    put_jobkey(w, m.job);
    w.u32(m.model_role); w.u64(m.model_version);
    w.b(m.fail_if_missing);
    w.u32(static_cast<uint32_t>(m.expected_param_ids.size()));
    for (uint64_t p : m.expected_param_ids) w.u64(p);
    w.b(m.promote);
    return b;
}
SealModelVersionRequest decode_SealModelVersionRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); SealModelVersionRequest m;
    m.job = get_jobkey(r);
    m.model_role = r.u32(); m.model_version = r.u64();
    m.fail_if_missing = r.b();
    uint32_t np = r.u32();
    m.expected_param_ids.reserve(np);
    for (uint32_t i = 0; i < np; ++i) m.expected_param_ids.push_back(r.u64());
    m.promote = r.b();
    return m;
}

std::vector<uint8_t> encode(const SealModelVersionResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u32(m.state); w.u64(m.tensor_count); w.u64(m.total_nbytes);
    w.u64(m.manifest_hash_lo); w.u64(m.manifest_hash_hi);
    return b;
}
SealModelVersionResponse decode_SealModelVersionResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); SealModelVersionResponse m;
    m.ok = r.b(); m.message = r.str();
    m.state = r.u32(); m.tensor_count = r.u64(); m.total_nbytes = r.u64();
    m.manifest_hash_lo = r.u64(); m.manifest_hash_hi = r.u64();
    return m;
}

// ---- GetLatestSealedVersion -----------------------------------------------
std::vector<uint8_t> encode(const GetLatestSealedVersionRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    put_jobkey(w, m.job); w.u32(m.model_role);
    return b;
}
GetLatestSealedVersionRequest decode_GetLatestSealedVersionRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); GetLatestSealedVersionRequest m;
    m.job = get_jobkey(r); m.model_role = r.u32();
    return m;
}

std::vector<uint8_t> encode(const GetLatestSealedVersionResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message); w.b(m.found); w.u64(m.model_version);
    return b;
}
GetLatestSealedVersionResponse decode_GetLatestSealedVersionResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); GetLatestSealedVersionResponse m;
    m.ok = r.b(); m.message = r.str(); m.found = r.b(); m.model_version = r.u64();
    return m;
}

// ---- GetManifest ----------------------------------------------------------
namespace {
void put_filter(Writer& w, const ManifestFilterWire& f) {
    w.b(f.pp_present);    w.u32(f.pp_rank);
    w.b(f.tp_present);    w.u32(f.tp_rank);
    w.b(f.ep_present);    w.u32(f.ep_rank);
    w.b(f.etp_present);   w.u32(f.etp_rank);
    w.b(f.expert_present); w.u32(static_cast<uint32_t>(f.expert_id));
}
ManifestFilterWire get_filter(Reader& r) {
    ManifestFilterWire f;
    f.pp_present = r.b();    f.pp_rank = r.u32();
    f.tp_present = r.b();    f.tp_rank = r.u32();
    f.ep_present = r.b();    f.ep_rank = r.u32();
    f.etp_present = r.b();   f.etp_rank = r.u32();
    f.expert_present = r.b(); f.expert_id = static_cast<int32_t>(r.u32());
    return f;
}
}  // namespace

std::vector<uint8_t> encode(const GetManifestRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    put_jobkey(w, m.job); w.u32(m.model_role); w.u64(m.model_version);
    put_filter(w, m.filter);
    return b;
}
GetManifestRequest decode_GetManifestRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); GetManifestRequest m;
    m.job = get_jobkey(r); m.model_role = r.u32(); m.model_version = r.u64();
    m.filter = get_filter(r);
    return m;
}

std::vector<uint8_t> encode(const GetManifestResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    put_jobkey(w, m.job); w.u32(m.model_role); w.u64(m.model_version); w.u32(m.state);
    w.u32(static_cast<uint32_t>(m.tensors.size()));
    for (const auto& t : m.tensors) {
        put_key(w, t.key);
        w.u64(t.object_id); w.u64(t.nbytes);
        w.u64(t.content_hash_lo); w.u64(t.content_hash_hi);
        w.str(t.location_hint);
    }
    return b;
}
GetManifestResponse decode_GetManifestResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); GetManifestResponse m;
    m.ok = r.b(); m.message = r.str();
    m.job = get_jobkey(r); m.model_role = r.u32(); m.model_version = r.u64();
    m.state = r.u32();
    uint32_t nt = r.u32();
    m.tensors.reserve(nt);
    for (uint32_t i = 0; i < nt; ++i) {
        TensorManifestEntryWire t;
        t.key = get_key(r);
        t.object_id = r.u64(); t.nbytes = r.u64();
        t.content_hash_lo = r.u64(); t.content_hash_hi = r.u64();
        t.location_hint = r.str();
        m.tensors.push_back(std::move(t));
    }
    return m;
}

// ---- PullTensor -----------------------------------------------------------
std::vector<uint8_t> encode(const PullTensorRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    put_jobkey(w, m.job); w.u32(m.model_role); w.u64(m.model_version);
    w.u64(m.object_id); w.u32(m.transport); w.str(m.target_descriptor);
    return b;
}
PullTensorRequest decode_PullTensorRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); PullTensorRequest m;
    m.job = get_jobkey(r); m.model_role = r.u32(); m.model_version = r.u64();
    m.object_id = r.u64(); m.transport = r.u32(); m.target_descriptor = r.str();
    return m;
}

std::vector<uint8_t> encode(const PullTensorResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u64(m.object_id); w.u64(m.nbytes); w.u32(m.transport);
    w.u64(m.content_hash_lo); w.u64(m.content_hash_hi);
    w.str(m.transport_metadata);
    return b;
}
PullTensorResponse decode_PullTensorResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); PullTensorResponse m;
    m.ok = r.b(); m.message = r.str();
    m.object_id = r.u64(); m.nbytes = r.u64(); m.transport = r.u32();
    m.content_hash_lo = r.u64(); m.content_hash_hi = r.u64();
    m.transport_metadata = r.str();
    return m;
}

}  // namespace offload
