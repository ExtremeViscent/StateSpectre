#include "wire.h"

#include "offload_abi.h"

namespace offload {

const char* op_name(OpCode op) {
    switch (op) {
        case OpCode::kInvalid:          return "Invalid";
        case OpCode::kRegisterRank:     return "RegisterRank";
        case OpCode::kHeartbeat:        return "Heartbeat";
        case OpCode::kRequestOffload:   return "RequestOffload";
        case OpCode::kMarkD2HSubmitted: return "MarkD2HSubmitted";
        case OpCode::kMarkD2HComplete:  return "MarkD2HComplete";
        case OpCode::kRequestPrefetch:  return "RequestPrefetch";
        case OpCode::kMarkH2DSubmitted: return "MarkH2DSubmitted";
        case OpCode::kMarkH2DComplete:  return "MarkH2DComplete";
        case OpCode::kReleaseLease:     return "ReleaseLease";
        case OpCode::kQueryLocation:    return "QueryLocation";
        case OpCode::kBatchComplete:    return "BatchComplete";
        case OpCode::kShutdown:         return "Shutdown";
        case OpCode::kGetStats:         return "GetStats";
        case OpCode::kRegisterJob:            return "RegisterJob";
        case OpCode::kRequestCanonicalEvict:  return "RequestCanonicalEvict";
        case OpCode::kCommitCanonicalObject:  return "CommitCanonicalObject";
        case OpCode::kSealModelVersion:       return "SealModelVersion";
        case OpCode::kGetLatestSealedVersion: return "GetLatestSealedVersion";
        case OpCode::kGetManifest:            return "GetManifest";
        case OpCode::kPullTensor:             return "PullTensor";
        case OpCode::kRequestCanonicalRestore: return "RequestCanonicalRestore";
        case OpCode::kReleaseCanonicalRestore: return "ReleaseCanonicalRestore";
        case OpCode::kDropCanonicalVersion:    return "DropCanonicalVersion";
    }
    return "Unknown";
}

// -------------------- framing --------------------

std::vector<uint8_t> make_frame(OpCode op, const std::vector<uint8_t>& body,
                                uint16_t flags) {
    if (body.size() > 0xFFFFFFFFull) throw WireError("frame body too large");
    std::vector<uint8_t> frame;
    frame.reserve(kFrameHeaderSize + body.size());
    Writer w(frame);
    w.u32(OFLD_RPC_MAGIC);
    w.u16(static_cast<uint16_t>(op));
    w.u16(flags);
    w.u32(static_cast<uint32_t>(body.size()));
    frame.insert(frame.end(), body.begin(), body.end());
    return frame;
}

FrameHeader parse_frame_header(const uint8_t* data, size_t len) {
    if (len < kFrameHeaderSize) throw WireError("short frame header");
    Reader r(data, len);
    FrameHeader h;
    h.magic = r.u32();
    if (h.magic != OFLD_RPC_MAGIC) throw WireError("bad frame magic");
    h.opcode = r.u16();
    h.flags = r.u16();
    h.payload_len = r.u32();
    return h;
}

// -------------------- per-message codecs --------------------
// Field order mirrors rpc/offload_daemon.proto exactly for readability.

// RegisterRankRequest
std::vector<uint8_t> encode(const RegisterRankRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u32(m.local_rank); w.u32(m.world_rank);
    w.u32(m.gpu_id); w.u32(m.numa_node); w.u64(m.pid); w.u64(m.capabilities);
    return b;
}
RegisterRankRequest decode_RegisterRankRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); RegisterRankRequest m;
    m.rank_id = r.u32(); m.local_rank = r.u32(); m.world_rank = r.u32();
    m.gpu_id = r.u32(); m.numa_node = r.u32(); m.pid = r.u64();
    m.capabilities = r.u64();
    return m;
}

// RegisterRankResponse
std::vector<uint8_t> encode(const RegisterRankResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u64(m.rank_epoch); w.u64(m.control_generation);
    w.u32(static_cast<uint32_t>(m.arenas.size()));
    for (const auto& a : m.arenas) {
        w.u64(a.arena_id); w.u32(a.numa_node); w.u32(a.kind); w.u64(a.size);
        w.u64(a.base_offset); w.u32(a.preferred_gpu); w.u64(a.flags);
        w.u32(a.fd_index); w.u32(a.registration_granularity);
        w.u32(a.allocation_granularity);
    }
    w.u32(m.control_fd_index); w.u64(m.control_size); w.u32(m.num_slots);
    w.u32(m.ring_index);
    return b;
}
RegisterRankResponse decode_RegisterRankResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); RegisterRankResponse m;
    m.ok = r.b(); m.message = r.str();
    m.rank_epoch = r.u64(); m.control_generation = r.u64();
    uint32_t na = r.u32();
    m.arenas.resize(na);
    for (auto& a : m.arenas) {
        a.arena_id = r.u64(); a.numa_node = r.u32(); a.kind = r.u32();
        a.size = r.u64(); a.base_offset = r.u64(); a.preferred_gpu = r.u32();
        a.flags = r.u64(); a.fd_index = r.u32();
        a.registration_granularity = r.u32(); a.allocation_granularity = r.u32();
    }
    m.control_fd_index = r.u32(); m.control_size = r.u64(); m.num_slots = r.u32();
    m.ring_index = r.u32();
    return m;
}

// HeartbeatRequest / Response
std::vector<uint8_t> encode(const HeartbeatRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.timestamp_ns);
    return b;
}
HeartbeatRequest decode_HeartbeatRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); HeartbeatRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.timestamp_ns = r.u64();
    return m;
}
std::vector<uint8_t> encode(const HeartbeatResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
HeartbeatResponse decode_HeartbeatResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); HeartbeatResponse m; m.ok = r.b(); m.message = r.str(); return m;
}

// RequestOffloadRequest / Response
std::vector<uint8_t> encode(const RequestOffloadRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch);
    w.u64(m.tensor_id); w.u64(m.version); w.u64(m.nbytes);
    w.u32(m.gpu_id); w.u32(m.numa_node); w.u32(m.alignment); w.u32(m.priority);
    w.u64(m.flags); w.str(m.debug_name);
    return b;
}
RequestOffloadRequest decode_RequestOffloadRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestOffloadRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.nbytes = r.u64();
    m.gpu_id = r.u32(); m.numa_node = r.u32(); m.alignment = r.u32();
    m.priority = r.u32(); m.flags = r.u64(); m.debug_name = r.str();
    return m;
}
std::vector<uint8_t> encode(const RequestOffloadResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u64(m.lease_id); w.u32(m.slot_id); w.u64(m.arena_id); w.u64(m.arena_offset);
    w.u64(m.capacity); w.u32(m.state); w.b(m.blocked);
    return b;
}
RequestOffloadResponse decode_RequestOffloadResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestOffloadResponse m;
    m.ok = r.b(); m.message = r.str();
    m.lease_id = r.u64(); m.slot_id = r.u32(); m.arena_id = r.u64();
    m.arena_offset = r.u64(); m.capacity = r.u64(); m.state = r.u32();
    m.blocked = r.b();
    return m;
}

// MarkD2HSubmitted
std::vector<uint8_t> encode(const MarkD2HSubmittedRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.lease_id);
    w.u64(m.tensor_id); w.u64(m.version); w.u32(m.slot_id);
    w.u64(m.submit_seq); w.u64(m.timestamp_ns);
    return b;
}
MarkD2HSubmittedRequest decode_MarkD2HSubmittedRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkD2HSubmittedRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.lease_id = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.slot_id = r.u32();
    m.submit_seq = r.u64(); m.timestamp_ns = r.u64();
    return m;
}
std::vector<uint8_t> encode(const MarkD2HSubmittedResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
MarkD2HSubmittedResponse decode_MarkD2HSubmittedResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkD2HSubmittedResponse m; m.ok = r.b(); m.message = r.str();
    return m;
}

// MarkD2HComplete
std::vector<uint8_t> encode(const MarkD2HCompleteRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.lease_id);
    w.u64(m.tensor_id); w.u64(m.version); w.u32(m.slot_id);
    w.u64(m.complete_seq); w.u64(m.timestamp_ns);
    return b;
}
MarkD2HCompleteRequest decode_MarkD2HCompleteRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkD2HCompleteRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.lease_id = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.slot_id = r.u32();
    m.complete_seq = r.u64(); m.timestamp_ns = r.u64();
    return m;
}
std::vector<uint8_t> encode(const MarkD2HCompleteResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message); w.b(m.latest_version); w.b(m.drain_enqueued);
    return b;
}
MarkD2HCompleteResponse decode_MarkD2HCompleteResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkD2HCompleteResponse m;
    m.ok = r.b(); m.message = r.str(); m.latest_version = r.b();
    m.drain_enqueued = r.b();
    return m;
}

// RequestPrefetch
std::vector<uint8_t> encode(const RequestPrefetchRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch);
    w.u64(m.tensor_id); w.u64(m.version); w.u64(m.nbytes);
    w.u32(m.gpu_id); w.u32(m.numa_node); w.u32(m.priority); w.u64(m.flags);
    return b;
}
RequestPrefetchRequest decode_RequestPrefetchRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestPrefetchRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.nbytes = r.u64();
    m.gpu_id = r.u32(); m.numa_node = r.u32(); m.priority = r.u32();
    m.flags = r.u64();
    return m;
}
std::vector<uint8_t> encode(const RequestPrefetchResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    w.u64(m.lease_id); w.u32(m.slot_id); w.u64(m.arena_id); w.u64(m.arena_offset);
    w.u64(m.nbytes); w.u32(m.state); w.b(m.ready); w.u64(m.version);
    return b;
}
RequestPrefetchResponse decode_RequestPrefetchResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); RequestPrefetchResponse m;
    m.ok = r.b(); m.message = r.str();
    m.lease_id = r.u64(); m.slot_id = r.u32(); m.arena_id = r.u64();
    m.arena_offset = r.u64(); m.nbytes = r.u64(); m.state = r.u32();
    m.ready = r.b(); m.version = r.u64();
    return m;
}

// MarkH2DSubmitted
std::vector<uint8_t> encode(const MarkH2DSubmittedRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.lease_id);
    w.u64(m.tensor_id); w.u64(m.version); w.u32(m.slot_id);
    w.u64(m.submit_seq); w.u64(m.timestamp_ns);
    return b;
}
MarkH2DSubmittedRequest decode_MarkH2DSubmittedRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkH2DSubmittedRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.lease_id = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.slot_id = r.u32();
    m.submit_seq = r.u64(); m.timestamp_ns = r.u64();
    return m;
}
std::vector<uint8_t> encode(const MarkH2DSubmittedResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
MarkH2DSubmittedResponse decode_MarkH2DSubmittedResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkH2DSubmittedResponse m; m.ok = r.b(); m.message = r.str();
    return m;
}

// MarkH2DComplete
std::vector<uint8_t> encode(const MarkH2DCompleteRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.lease_id);
    w.u64(m.tensor_id); w.u64(m.version); w.u32(m.slot_id);
    w.b(m.keep_pinned); w.u64(m.complete_seq); w.u64(m.timestamp_ns);
    return b;
}
MarkH2DCompleteRequest decode_MarkH2DCompleteRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkH2DCompleteRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.lease_id = r.u64();
    m.tensor_id = r.u64(); m.version = r.u64(); m.slot_id = r.u32();
    m.keep_pinned = r.b(); m.complete_seq = r.u64(); m.timestamp_ns = r.u64();
    return m;
}
std::vector<uint8_t> encode(const MarkH2DCompleteResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
MarkH2DCompleteResponse decode_MarkH2DCompleteResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); MarkH2DCompleteResponse m; m.ok = r.b(); m.message = r.str();
    return m;
}

// ReleaseLease
std::vector<uint8_t> encode(const ReleaseLeaseRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch); w.u64(m.lease_id); w.u32(m.slot_id);
    w.str(m.reason);
    return b;
}
ReleaseLeaseRequest decode_ReleaseLeaseRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); ReleaseLeaseRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64(); m.lease_id = r.u64();
    m.slot_id = r.u32(); m.reason = r.str();
    return m;
}
std::vector<uint8_t> encode(const ReleaseLeaseResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
ReleaseLeaseResponse decode_ReleaseLeaseResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); ReleaseLeaseResponse m; m.ok = r.b(); m.message = r.str();
    return m;
}

// LocationQuery
std::vector<uint8_t> encode(const LocationQueryRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u64(m.tensor_id); w.u64(m.version);
    return b;
}
LocationQueryRequest decode_LocationQueryRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); LocationQueryRequest m;
    m.tensor_id = r.u64(); m.version = r.u64();
    return m;
}
std::vector<uint8_t> encode(const LocationQueryResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message); w.u32(m.location_kind); w.u32(m.slot_id);
    w.u64(m.nbytes); w.u64(m.version);
    return b;
}
LocationQueryResponse decode_LocationQueryResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); LocationQueryResponse m;
    m.ok = r.b(); m.message = r.str(); m.location_kind = r.u32();
    m.slot_id = r.u32(); m.nbytes = r.u64(); m.version = r.u64();
    return m;
}

// BatchComplete
std::vector<uint8_t> encode(const BatchCompleteRequest& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.u32(m.rank_id); w.u64(m.rank_epoch);
    w.u32(static_cast<uint32_t>(m.completions.size()));
    for (const auto& c : m.completions) {
        w.u32(c.event_type); w.u64(c.lease_id); w.u64(c.tensor_id);
        w.u64(c.version); w.u32(c.slot_id); w.u64(c.seq); w.u64(c.timestamp_ns);
        w.b(c.keep_pinned);
    }
    return b;
}
BatchCompleteRequest decode_BatchCompleteRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); BatchCompleteRequest m;
    m.rank_id = r.u32(); m.rank_epoch = r.u64();
    uint32_t k = r.u32();
    m.completions.resize(k);
    for (auto& c : m.completions) {
        c.event_type = r.u32(); c.lease_id = r.u64(); c.tensor_id = r.u64();
        c.version = r.u64(); c.slot_id = r.u32(); c.seq = r.u64();
        c.timestamp_ns = r.u64(); c.keep_pinned = r.b();
    }
    return m;
}
std::vector<uint8_t> encode(const BatchCompleteResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message); w.u32(m.accepted); w.u32(m.rejected);
    return b;
}
BatchCompleteResponse decode_BatchCompleteResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); BatchCompleteResponse m;
    m.ok = r.b(); m.message = r.str(); m.accepted = r.u32(); m.rejected = r.u32();
    return m;
}

// GetStats
std::vector<uint8_t> encode(const GetStatsRequest& m) {
    std::vector<uint8_t> b; Writer w(b); w.u32(m.rank_id); return b;
}
GetStatsRequest decode_GetStatsRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); GetStatsRequest m; m.rank_id = r.u32(); return m;
}
std::vector<uint8_t> encode(const GetStatsResponse& m) {
    std::vector<uint8_t> b; Writer w(b);
    w.b(m.ok); w.str(m.message);
    uint32_t k = static_cast<uint32_t>(m.keys.size());
    w.u32(k);
    for (uint32_t i = 0; i < k; ++i) { w.str(m.keys[i]); w.u64(m.values[i]); }
    return b;
}
GetStatsResponse decode_GetStatsResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); GetStatsResponse m;
    m.ok = r.b(); m.message = r.str();
    uint32_t k = r.u32();
    m.keys.resize(k); m.values.resize(k);
    for (uint32_t i = 0; i < k; ++i) { m.keys[i] = r.str(); m.values[i] = r.u64(); }
    return m;
}

// Shutdown
std::vector<uint8_t> encode(const ShutdownRequest& m) {
    std::vector<uint8_t> b; Writer w(b); w.u64(m.token); return b;
}
ShutdownRequest decode_ShutdownRequest(const uint8_t* d, size_t n) {
    Reader r(d, n); ShutdownRequest m; m.token = r.u64(); return m;
}
std::vector<uint8_t> encode(const ShutdownResponse& m) {
    std::vector<uint8_t> b; Writer w(b); w.b(m.ok); w.str(m.message); return b;
}
ShutdownResponse decode_ShutdownResponse(const uint8_t* d, size_t n) {
    Reader r(d, n); ShutdownResponse m; m.ok = r.b(); m.message = r.str(); return m;
}

}  // namespace offload
