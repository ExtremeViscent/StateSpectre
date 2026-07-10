#include "uds.h"

#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "util.h"

namespace offload {

namespace {
constexpr int kMaxFds = 32;

void write_all(int sock, const uint8_t* buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = ::send(sock, buf + off, len - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("send failed: ") + strerror(errno));
        }
        if (n == 0) throw std::runtime_error("send returned 0 (peer closed)");
        off += static_cast<size_t>(n);
    }
}

// Read exactly len bytes. Returns false on clean EOF at a message boundary
// (i.e. zero bytes read on the very first recv), true on full read.
bool read_all(int sock, uint8_t* buf, size_t len, bool* eof_at_start) {
    size_t off = 0;
    if (eof_at_start) *eof_at_start = false;
    while (off < len) {
        ssize_t n = ::recv(sock, buf + off, len - off, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("recv failed: ") + strerror(errno));
        }
        if (n == 0) {
            if (off == 0 && eof_at_start) { *eof_at_start = true; return false; }
            throw std::runtime_error("recv EOF mid-message");
        }
        off += static_cast<size_t>(n);
    }
    return true;
}
}  // namespace

void send_frame(int sock, const std::vector<uint8_t>& frame,
                const std::vector<int>& fds) {
    if (frame.size() < kFrameHeaderSize)
        throw std::runtime_error("send_frame: frame smaller than header");

    if (fds.empty()) {
        write_all(sock, frame.data(), frame.size());
        return;
    }
    if (fds.size() > kMaxFds) throw std::runtime_error("too many fds for one frame");

    // Send the header (at least 1 byte) with the fds attached as ancillary
    // data, then stream the remainder normally. We attach fds to the first
    // sendmsg carrying the 12-byte header so the receiver reads them alongside
    // the header parse.
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));

    struct iovec iov;
    iov.iov_base = const_cast<uint8_t*>(frame.data());
    iov.iov_len = kFrameHeaderSize;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;

    char cbuf[CMSG_SPACE(sizeof(int) * kMaxFds)];
    memset(cbuf, 0, sizeof(cbuf));
    msg.msg_control = cbuf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * fds.size());

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
    memcpy(CMSG_DATA(cmsg), fds.data(), sizeof(int) * fds.size());

    // sendmsg the header + ancillary (retry on EINTR / short header write).
    size_t hdr_sent = 0;
    while (hdr_sent < kFrameHeaderSize) {
        iov.iov_base = const_cast<uint8_t*>(frame.data()) + hdr_sent;
        iov.iov_len = kFrameHeaderSize - hdr_sent;
        // Only attach control message on the first successful sendmsg.
        ssize_t n = ::sendmsg(sock, &msg, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("sendmsg failed: ") + strerror(errno));
        }
        hdr_sent += static_cast<size_t>(n);
        // After the first send, drop the control data (fds already handed off).
        msg.msg_control = nullptr;
        msg.msg_controllen = 0;
    }

    // Stream the payload (if any) after the header.
    if (frame.size() > kFrameHeaderSize) {
        write_all(sock, frame.data() + kFrameHeaderSize,
                  frame.size() - kFrameHeaderSize);
    }
}

bool recv_frame(int sock, OpCode* op, std::vector<uint8_t>* payload,
                std::vector<int>* fds) {
    if (fds) fds->clear();
    if (payload) payload->clear();

    // Receive the 12-byte header along with any ancillary fds.
    uint8_t hdr[kFrameHeaderSize];
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    struct iovec iov;
    iov.iov_base = hdr;
    iov.iov_len = kFrameHeaderSize;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    char cbuf[CMSG_SPACE(sizeof(int) * kMaxFds)];
    memset(cbuf, 0, sizeof(cbuf));
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);

    size_t got = 0;
    bool first = true;
    while (got < kFrameHeaderSize) {
        iov.iov_base = hdr + got;
        iov.iov_len = kFrameHeaderSize - got;
        ssize_t n = ::recvmsg(sock, &msg, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("recvmsg failed: ") + strerror(errno));
        }
        if (n == 0) {
            if (got == 0 && first) return false;  // clean EOF at boundary
            throw std::runtime_error("recvmsg EOF mid-header");
        }
        // Harvest any fds from this recvmsg (they arrive with the first bytes).
        if (fds) {
            for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr;
                 c = CMSG_NXTHDR(&msg, c)) {
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
                    int cnt = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                    const int* fdp = reinterpret_cast<const int*>(CMSG_DATA(c));
                    for (int i = 0; i < cnt; ++i) fds->push_back(fdp[i]);
                }
            }
        }
        got += static_cast<size_t>(n);
        // Only the first recvmsg carries ancillary space; disable afterward.
        msg.msg_control = nullptr;
        msg.msg_controllen = 0;
        first = false;
    }

    FrameHeader fh = parse_frame_header(hdr, kFrameHeaderSize);
    if (op) *op = static_cast<OpCode>(fh.opcode);

    if (fh.payload_len > (256u << 20))
        throw std::runtime_error("frame payload absurdly large");

    if (payload) {
        payload->resize(fh.payload_len);
        if (fh.payload_len > 0) {
            bool eof = false;
            read_all(sock, payload->data(), fh.payload_len, &eof);
            if (eof) throw std::runtime_error("EOF before payload");
        }
    }
    return true;
}

int uds_listen(const std::string& path, int backlog) {
    if (path.size() >= sizeof(((struct sockaddr_un*)0)->sun_path))
        throw std::runtime_error("socket path too long");

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket() failed: ") + strerror(errno));

    ::unlink(path.c_str());  // remove stale socket

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(std::string("bind(") + path + ") failed: " + strerror(e));
    }
    if (::listen(fd, backlog) < 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error(std::string("listen failed: ") + strerror(e));
    }
    OFLD_INFO("uds", "listening on %s (fd=%d)", path.c_str(), fd);
    return fd;
}

int uds_accept(int listen_fd) {
    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
        if (errno == EINTR || errno == EAGAIN) return -1;
        throw std::runtime_error(std::string("accept failed: ") + strerror(errno));
    }
    return fd;
}

int uds_connect(const std::string& path, int timeout_ms) {
    if (path.size() >= sizeof(((struct sockaddr_un*)0)->sun_path))
        throw std::runtime_error("socket path too long");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    const int retry_ms = 25;
    int waited = 0;
    for (;;) {
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error(std::string("socket() failed: ") + strerror(errno));
        if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        int e = errno;
        ::close(fd);
        if ((e == ENOENT || e == ECONNREFUSED) && waited < timeout_ms) {
            struct timespec ts { retry_ms / 1000, (retry_ms % 1000) * 1000000L };
            nanosleep(&ts, nullptr);
            waited += retry_ms;
            continue;
        }
        throw std::runtime_error(std::string("connect(") + path + ") failed: " + strerror(e));
    }
}

void close_fd(int fd) {
    if (fd >= 0) ::close(fd);
}

}  // namespace offload
