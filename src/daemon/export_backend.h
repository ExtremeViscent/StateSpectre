// Export transport backends for canonical object pull (PullTensor).
//
// The backend is deliberately dumb: it receives a host buffer + an opaque
// target descriptor and ships the bytes. It knows nothing about PyTorch,
// canonical keys, or slots (LIBFABRIC_EXPORT_BACKEND.md: "the libfabric sender
// is a transport backend ... it receives object/export buffer descriptors").
//
// Transports (TransportKind in offload_canonical.proto):
//   TCP  (1): connect to "host:port" in target_descriptor, send framed bytes.
//   FILE (2): write bytes to the path in target_descriptor (debug/local).
//   LIBFABRIC_SEND (3): north_comm request-triggered send. Compiled only when
//                       OFLD_WITH_LIBFABRIC is defined; otherwise falls back to
//                       the configured fallback transport.

#pragma once

#include <cstdint>
#include <string>

namespace offload {

struct ExportRequest {
    uint64_t object_id = 0;
    uint64_t nbytes = 0;
    uint32_t transport = 0;            // TransportKind
    std::string target_descriptor;     // "host:port" | path | libfabric desc
    uint64_t content_hash_lo = 0;
    uint64_t content_hash_hi = 0;
};

struct ExportResult {
    bool ok = false;
    std::string message;
    uint32_t transport = 0;            // transport actually used
    uint64_t bytes_sent = 0;
    uint64_t duration_ns = 0;
    std::string transport_metadata;    // transfer id / staging id / rkey
};

// Ship `nbytes` bytes from `data` per req.transport. If the requested transport
// is unavailable (e.g. libfabric not compiled in), falls back to
// `fallback_transport`. Never throws; errors come back in ExportResult.
ExportResult export_send(const ExportRequest& req, const void* data,
                         uint64_t nbytes, uint32_t fallback_transport);

}  // namespace offload
