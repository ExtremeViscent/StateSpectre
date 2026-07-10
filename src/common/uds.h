// AF_UNIX SOCK_STREAM transport with length-framed messages and SCM_RIGHTS
// file-descriptor passing.
//
// A single request/response exchange = one frame each direction. File
// descriptors (arena + control shm fds) are attached to the RegisterRank
// response frame as ancillary data. All other frames carry no fds.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "wire.h"

namespace offload {

// Send a full frame (header+payload) plus optional fds over a connected socket.
// Blocks until fully written. Throws std::runtime_error on fatal socket error.
// fds are attached via SCM_RIGHTS on the first sendmsg; the payload may span
// multiple write() calls but fds ride only the first.
void send_frame(int sock, const std::vector<uint8_t>& frame,
                const std::vector<int>& fds = {});

// Receive exactly one frame. Returns the opcode via *op, the payload bytes via
// *payload, and any received fds via *fds (caller owns/closes them).
// Returns false on clean EOF (peer closed with nothing pending), true on a
// full frame. Throws on protocol/socket error.
bool recv_frame(int sock, OpCode* op, std::vector<uint8_t>* payload,
                std::vector<int>* fds);

// --- server side ---
// Create, bind, and listen on a Unix domain socket at path. Unlinks any stale
// socket file first. Returns the listening fd. Throws on failure.
int uds_listen(const std::string& path, int backlog = 128);

// Create, bind, and listen on a TCP socket on the given host-order port (all
// interfaces, SO_REUSEADDR). Returns the listening fd. Throws on failure. Used
// by the v2 canonical control endpoint so remote rollout engines can reach the
// manifest/pull RPCs. Frames on this fd carry NO SCM_RIGHTS fds.
int tcp_listen(uint16_t port, int backlog = 128);

// Accept one connection. Returns connected fd (>=0), or -1 if interrupted.
int uds_accept(int listen_fd);

// --- client side ---
// Connect to a Unix domain socket. Retries up to timeout_ms if the daemon is
// not yet listening. Returns connected fd. Throws on failure/timeout.
int uds_connect(const std::string& path, int timeout_ms = 5000);

void close_fd(int fd);

}  // namespace offload
