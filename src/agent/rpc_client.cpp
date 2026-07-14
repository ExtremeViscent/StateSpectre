#include "rpc_client.h"

#include <string>

#include "uds.h"
#include "wire.h"
#include "wire_v2.h"

namespace offload {

namespace {

// Perform one request/response round-trip on `sock` for opcode `op`, using the
// already-encoded request body. Sends the request frame, receives exactly one
// reply frame, asserts the reply opcode matches, and returns the raw payload.
// `send_fds` (usually empty) rides the request frame; `out_fds` (may be null)
// collects any fds attached to the reply. Throws std::runtime_error on any
// transport or protocol error.
std::vector<uint8_t> round_trip(int sock, OpCode op,
                                const std::vector<uint8_t>& body,
                                std::vector<int>* out_fds) {
    send_frame(sock, make_frame(op, body), {});

    OpCode reply_op = OpCode::kInvalid;
    std::vector<uint8_t> payload;
    bool ok = recv_frame(sock, &reply_op, &payload, out_fds);
    if (!ok) {
        throw std::runtime_error(std::string("rpc: connection closed while awaiting reply to ") +
                                 op_name(op));
    }
    if (reply_op != op) {
        throw std::runtime_error(std::string("rpc: reply opcode mismatch: sent ") +
                                 op_name(op) + " got " + op_name(reply_op));
    }
    return payload;
}

}  // namespace

RpcClient::RpcClient(int connected_sock) : sock_(connected_sock) {}

RpcClient::~RpcClient() {
    if (sock_ >= 0) {
        close_fd(sock_);
        sock_ = -1;
    }
}

RegisterRankResponse RpcClient::register_rank(const RegisterRankRequest& req,
                                              std::vector<int>* out_fds) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kRegisterRank, encode(req), out_fds);
    return decode_RegisterRankResponse(payload.data(), payload.size());
}

RequestOffloadResponse RpcClient::request_offload(const RequestOffloadRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kRequestOffload, encode(req), nullptr);
    return decode_RequestOffloadResponse(payload.data(), payload.size());
}

MarkD2HSubmittedResponse RpcClient::mark_d2h_submitted(const MarkD2HSubmittedRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kMarkD2HSubmitted, encode(req), nullptr);
    return decode_MarkD2HSubmittedResponse(payload.data(), payload.size());
}

MarkD2HCompleteResponse RpcClient::mark_d2h_complete(const MarkD2HCompleteRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kMarkD2HComplete, encode(req), nullptr);
    return decode_MarkD2HCompleteResponse(payload.data(), payload.size());
}

RequestPrefetchResponse RpcClient::request_prefetch(const RequestPrefetchRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kRequestPrefetch, encode(req), nullptr);
    return decode_RequestPrefetchResponse(payload.data(), payload.size());
}

MarkH2DSubmittedResponse RpcClient::mark_h2d_submitted(const MarkH2DSubmittedRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kMarkH2DSubmitted, encode(req), nullptr);
    return decode_MarkH2DSubmittedResponse(payload.data(), payload.size());
}

MarkH2DCompleteResponse RpcClient::mark_h2d_complete(const MarkH2DCompleteRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kMarkH2DComplete, encode(req), nullptr);
    return decode_MarkH2DCompleteResponse(payload.data(), payload.size());
}

ReleaseLeaseResponse RpcClient::release_lease(const ReleaseLeaseRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kReleaseLease, encode(req), nullptr);
    return decode_ReleaseLeaseResponse(payload.data(), payload.size());
}

LocationQueryResponse RpcClient::query_location(const LocationQueryRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kQueryLocation, encode(req), nullptr);
    return decode_LocationQueryResponse(payload.data(), payload.size());
}

BatchCompleteResponse RpcClient::batch_complete(const BatchCompleteRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kBatchComplete, encode(req), nullptr);
    return decode_BatchCompleteResponse(payload.data(), payload.size());
}

HeartbeatResponse RpcClient::heartbeat(const HeartbeatRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kHeartbeat, encode(req), nullptr);
    return decode_HeartbeatResponse(payload.data(), payload.size());
}

GetStatsResponse RpcClient::get_stats(const GetStatsRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<uint8_t> payload =
        round_trip(sock_, OpCode::kGetStats, encode(req), nullptr);
    return decode_GetStatsResponse(payload.data(), payload.size());
}

// ---- v2 canonical model-state ops ----
RegisterJobResponse RpcClient::register_job(const RegisterJobRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kRegisterJob, encode(req), nullptr);
    return decode_RegisterJobResponse(p.data(), p.size());
}
RequestCanonicalEvictResponse RpcClient::request_canonical_evict(
    const RequestCanonicalEvictRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kRequestCanonicalEvict, encode(req), nullptr);
    return decode_RequestCanonicalEvictResponse(p.data(), p.size());
}
CommitCanonicalObjectResponse RpcClient::commit_canonical_object(
    const CommitCanonicalObjectRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kCommitCanonicalObject, encode(req), nullptr);
    return decode_CommitCanonicalObjectResponse(p.data(), p.size());
}
SealModelVersionResponse RpcClient::seal_model_version(
    const SealModelVersionRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kSealModelVersion, encode(req), nullptr);
    return decode_SealModelVersionResponse(p.data(), p.size());
}
GetLatestSealedVersionResponse RpcClient::get_latest_sealed_version(
    const GetLatestSealedVersionRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kGetLatestSealedVersion, encode(req), nullptr);
    return decode_GetLatestSealedVersionResponse(p.data(), p.size());
}
GetManifestResponse RpcClient::get_manifest(const GetManifestRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kGetManifest, encode(req), nullptr);
    return decode_GetManifestResponse(p.data(), p.size());
}
PullTensorResponse RpcClient::pull_tensor(const PullTensorRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kPullTensor, encode(req), nullptr);
    return decode_PullTensorResponse(p.data(), p.size());
}
RequestCanonicalRestoreResponse RpcClient::request_canonical_restore(
    const RequestCanonicalRestoreRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kRequestCanonicalRestore, encode(req), nullptr);
    return decode_RequestCanonicalRestoreResponse(p.data(), p.size());
}
ReleaseCanonicalRestoreResponse RpcClient::release_canonical_restore(
    const ReleaseCanonicalRestoreRequest& req) {
    std::lock_guard<std::mutex> lg(mu_);
    auto p = round_trip(sock_, OpCode::kReleaseCanonicalRestore, encode(req), nullptr);
    return decode_ReleaseCanonicalRestoreResponse(p.data(), p.size());
}

}  // namespace offload
