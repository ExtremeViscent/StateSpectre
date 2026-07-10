// Export transport backends. See export_backend.h.

#include "export_backend.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "util.h"

namespace offload {

namespace {
constexpr const char* TAG = "export";

// TransportKind values (mirror offload_canonical.proto).
enum { TK_UNKNOWN = 0, TK_TCP = 1, TK_FILE = 2, TK_LIBFABRIC_SEND = 3 };

// A tiny self-describing header so a debug receiver can frame the payload.
// [magic "OFEX"][u32 version=1][u64 object_id][u64 nbytes][u64 hlo][u64 hhi]
constexpr uint32_t kExportMagic = 0x4F464558u;  // 'OFEX'
struct ExportHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t object_id;
    uint64_t nbytes;
    uint64_t hlo;
    uint64_t hhi;
};

bool write_all(int fd, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::write(fd, b + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (w == 0) return false;
        off += static_cast<size_t>(w);
    }
    return true;
}

// Parse "host:port"; returns false if malformed.
bool split_hostport(const std::string& s, std::string* host, std::string* port) {
    auto pos = s.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
    *host = s.substr(0, pos);
    *port = s.substr(pos + 1);
    return true;
}

int tcp_connect(const std::string& host, const std::string& port) {
    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        return -1;
    int fd = -1;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    return fd;
}

ExportResult send_tcp(const ExportRequest& req, const void* data, uint64_t nbytes) {
    ExportResult r;
    r.transport = TK_TCP;
    std::string host, port;
    if (!split_hostport(req.target_descriptor, &host, &port)) {
        r.message = "bad tcp target (want host:port)";
        return r;
    }
    int fd = tcp_connect(host, port);
    if (fd < 0) { r.message = "tcp connect failed: " + std::string(std::strerror(errno)); return r; }
    ExportHeader h{kExportMagic, 1, req.object_id, nbytes, req.content_hash_lo,
                   req.content_hash_hi};
    bool ok = write_all(fd, &h, sizeof(h)) && write_all(fd, data, nbytes);
    ::close(fd);
    if (!ok) { r.message = "tcp write failed"; return r; }
    r.ok = true;
    r.bytes_sent = nbytes;
    r.transport_metadata = "tcp:" + req.target_descriptor;
    r.message = "ok";
    return r;
}

ExportResult send_file(const ExportRequest& req, const void* data, uint64_t nbytes) {
    ExportResult r;
    r.transport = TK_FILE;
    int fd = ::open(req.target_descriptor.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { r.message = "open failed: " + std::string(std::strerror(errno)); return r; }
    ExportHeader h{kExportMagic, 1, req.object_id, nbytes, req.content_hash_lo,
                   req.content_hash_hi};
    bool ok = write_all(fd, &h, sizeof(h)) && write_all(fd, data, nbytes);
    ::close(fd);
    if (!ok) { r.message = "file write failed"; return r; }
    r.ok = true;
    r.bytes_sent = nbytes;
    r.transport_metadata = "file:" + req.target_descriptor;
    r.message = "ok";
    return r;
}

}  // namespace

#ifdef OFLD_WITH_LIBFABRIC
// north_comm adapter, defined in export_libfabric.cpp (namespace-scope, so it
// links across TUs). Only declared/used when -DOFLD_WITH_LIBFABRIC is set.
ExportResult send_libfabric(const ExportRequest& req, const void* data,
                            uint64_t nbytes);
#endif

ExportResult export_send(const ExportRequest& req, const void* data,
                         uint64_t nbytes, uint32_t fallback_transport) {
    uint64_t t0 = now_mono_ns();
    uint32_t tk = req.transport;

    ExportResult r;
    switch (tk) {
        case TK_TCP:  r = send_tcp(req, data, nbytes); break;
        case TK_FILE: r = send_file(req, data, nbytes); break;
        case TK_LIBFABRIC_SEND:
#ifdef OFLD_WITH_LIBFABRIC
            r = send_libfabric(req, data, nbytes);
            break;
#else
            OFLD_WARN(TAG, "libfabric transport not compiled in; "
                            "falling back to transport %u", fallback_transport);
            {
                ExportRequest fb = req;
                fb.transport = fallback_transport;
                r = export_send(fb, data, nbytes, TK_TCP);
                r.duration_ns = now_mono_ns() - t0;
                return r;
            }
#endif
        default:
            r.message = "unknown transport";
            r.transport = tk;
            break;
    }
    r.duration_ns = now_mono_ns() - t0;
    return r;
}

}  // namespace offload
