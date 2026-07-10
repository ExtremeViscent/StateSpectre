// Thin synchronous RPC client over the AF_UNIX control-plane transport.
//
// One RpcClient wraps a single connected socket. Every method performs a full
// request/response round-trip while holding mu_, so the socket is used by at
// most one exchange at a time. This makes it safe for the agent's user threads
// (evict/restore), the heartbeat thread, and the progress thread to share one
// client concurrently: the mutex serializes whole round-trips, which is exactly
// the correctness boundary the control plane needs.
//
// RegisterRank is special: its reply carries the arena + control fds via
// SCM_RIGHTS, so it exposes an out-param for the received fds. All other ops
// carry no fds.

#pragma once

#include <mutex>
#include <stdexcept>
#include <vector>

#include "protocol.h"

namespace offload {

class RpcClient {
 public:
    // Takes ownership of an already-connected socket fd; closes it on destroy.
    explicit RpcClient(int connected_sock);
    ~RpcClient();

    RpcClient(const RpcClient&) = delete;
    RpcClient& operator=(const RpcClient&) = delete;

    // RegisterRank returns fds out-of-band via SCM_RIGHTS. The received fds are
    // appended (in arrival order) to *out_fds; the caller owns/closes them.
    RegisterRankResponse register_rank(const RegisterRankRequest& req,
                                       std::vector<int>* out_fds);

    // Generic request/response ops (no fds).
    RequestOffloadResponse    request_offload(const RequestOffloadRequest& req);
    MarkD2HSubmittedResponse  mark_d2h_submitted(const MarkD2HSubmittedRequest& req);
    MarkD2HCompleteResponse   mark_d2h_complete(const MarkD2HCompleteRequest& req);
    RequestPrefetchResponse   request_prefetch(const RequestPrefetchRequest& req);
    MarkH2DSubmittedResponse  mark_h2d_submitted(const MarkH2DSubmittedRequest& req);
    MarkH2DCompleteResponse   mark_h2d_complete(const MarkH2DCompleteRequest& req);
    ReleaseLeaseResponse      release_lease(const ReleaseLeaseRequest& req);
    LocationQueryResponse     query_location(const LocationQueryRequest& req);
    BatchCompleteResponse     batch_complete(const BatchCompleteRequest& req);
    HeartbeatResponse         heartbeat(const HeartbeatRequest& req);
    GetStatsResponse          get_stats(const GetStatsRequest& req);

    int sock() const { return sock_; }

 private:
    std::mutex mu_;  // serialize each full request/response round-trip
    int sock_;
};

}  // namespace offload
