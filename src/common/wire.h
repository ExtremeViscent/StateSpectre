// Byte-level codec + frame (de)serialization for the control-plane protocol.
//
// Encoding is little-endian, length-prefixed. Every request/response type has
// encode()/decode() free functions. A frame is:
//   [u32 magic][u16 opcode][u16 flags][u32 payload_len][payload bytes...]
// The payload is the encoded message body.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "protocol.h"

namespace offload {

// Thrown on malformed / truncated wire data.
class WireError : public std::runtime_error {
 public:
    explicit WireError(const std::string& m) : std::runtime_error(m) {}
};

// -------- primitive writer --------
class Writer {
 public:
    explicit Writer(std::vector<uint8_t>& buf) : buf_(buf) {}

    void u8(uint8_t v)  { buf_.push_back(v); }
    void b(bool v)      { buf_.push_back(v ? 1 : 0); }
    void u16(uint16_t v) { put(&v, sizeof(v)); }
    void u32(uint32_t v) { put(&v, sizeof(v)); }
    void u64(uint64_t v) { put(&v, sizeof(v)); }
    void str(const std::string& s) {
        if (s.size() > 0xFFFFFFFFull) throw WireError("string too long");
        u32(static_cast<uint32_t>(s.size()));
        buf_.insert(buf_.end(), s.begin(), s.end());
    }

 private:
    void put(const void* p, size_t n) {
        const uint8_t* c = static_cast<const uint8_t*>(p);
        buf_.insert(buf_.end(), c, c + n);
    }
    std::vector<uint8_t>& buf_;
};

// -------- primitive reader --------
class Reader {
 public:
    Reader(const uint8_t* data, size_t len) : p_(data), end_(data + len) {}

    uint8_t u8() { need(1); return *p_++; }
    bool b()     { return u8() != 0; }
    uint16_t u16() { uint16_t v; take(&v, sizeof(v)); return v; }
    uint32_t u32() { uint32_t v; take(&v, sizeof(v)); return v; }
    uint64_t u64() { uint64_t v; take(&v, sizeof(v)); return v; }
    std::string str() {
        uint32_t n = u32();
        need(n);
        std::string s(reinterpret_cast<const char*>(p_), n);
        p_ += n;
        return s;
    }
    bool empty() const { return p_ >= end_; }
    size_t remaining() const { return static_cast<size_t>(end_ - p_); }

 private:
    void need(size_t n) { if (remaining() < n) throw WireError("truncated wire message"); }
    void take(void* dst, size_t n) { need(n); std::memcpy(dst, p_, n); p_ += n; }
    const uint8_t* p_;
    const uint8_t* end_;
};

// -------- frame header --------
struct FrameHeader {
    uint32_t magic = OFLD_RPC_MAGIC;
    uint16_t opcode = 0;
    uint16_t flags = 0;
    uint32_t payload_len = 0;
};
static constexpr size_t kFrameHeaderSize = 12;  // 4 + 2 + 2 + 4

// Build a full frame (header + payload) for opcode from an already-encoded body.
std::vector<uint8_t> make_frame(OpCode op, const std::vector<uint8_t>& body,
                                uint16_t flags = 0);

// Parse just the 12-byte header. Throws WireError on bad magic/size.
FrameHeader parse_frame_header(const uint8_t* data, size_t len);

// -------- per-message encode/decode (payload body only) --------
#define OFLD_CODEC(T)                                        \
    std::vector<uint8_t> encode(const T& m);                 \
    T decode_##T(const uint8_t* data, size_t len);

OFLD_CODEC(RegisterRankRequest)
OFLD_CODEC(RegisterRankResponse)
OFLD_CODEC(HeartbeatRequest)
OFLD_CODEC(HeartbeatResponse)
OFLD_CODEC(RequestOffloadRequest)
OFLD_CODEC(RequestOffloadResponse)
OFLD_CODEC(MarkD2HSubmittedRequest)
OFLD_CODEC(MarkD2HSubmittedResponse)
OFLD_CODEC(MarkD2HCompleteRequest)
OFLD_CODEC(MarkD2HCompleteResponse)
OFLD_CODEC(RequestPrefetchRequest)
OFLD_CODEC(RequestPrefetchResponse)
OFLD_CODEC(MarkH2DSubmittedRequest)
OFLD_CODEC(MarkH2DSubmittedResponse)
OFLD_CODEC(MarkH2DCompleteRequest)
OFLD_CODEC(MarkH2DCompleteResponse)
OFLD_CODEC(ReleaseLeaseRequest)
OFLD_CODEC(ReleaseLeaseResponse)
OFLD_CODEC(LocationQueryRequest)
OFLD_CODEC(LocationQueryResponse)
OFLD_CODEC(BatchCompleteRequest)
OFLD_CODEC(BatchCompleteResponse)
OFLD_CODEC(GetStatsRequest)
OFLD_CODEC(GetStatsResponse)
OFLD_CODEC(ShutdownRequest)
OFLD_CODEC(ShutdownResponse)

#undef OFLD_CODEC

}  // namespace offload
