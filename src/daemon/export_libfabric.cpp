// north_comm / libfabric export backend adapter.
//
// Compiled ONLY when -DOFLD_WITH_LIBFABRIC is set and the north_comm headers +
// libnorth_comm are on the include/link path (see CMakeLists NORTH_COMM_DIR).
// When not compiled in, export_backend.cpp's LIBFABRIC_SEND path falls back to
// TCP, so the rest of the system builds and runs unchanged.
//
// Per LIBFABRIC_EXPORT_BACKEND.md this backend is intentionally dumb: it takes
// a host buffer + an opaque target descriptor and performs a request-triggered
// send. It does not know about tensors, canonical keys, or slots.
//
// Target descriptor format (opaque at the RPC layer; agreed with the rollout
// client): "<nic_name>|<ib_address_hex>". The rollout engine obtains the
// trainer endpoint's IbAddress out of band (it created a listener) and passes
// it here; the daemon connects and sends the staged export buffer.
//
// north_comm public surface used (from include/north_comm/, verified against
// the repo): Network, LibfabricEndpointNode, LibfabricEndpoint, IbAddress,
// Buffer::Alloc<Bytes>, Network::RegisterSendMemory. See north_comm/README.md
// and benchmark/cpu_send_recv.py for the reference send sequence.

#include "export_backend.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "util.h"

#include <north_comm/buffer.h>
#include <north_comm/libfabric/endpoint.h>
#include <north_comm/libfabric/network.h>
#include <north_comm/libfabric/ib_address.h>

namespace offload {
namespace {
constexpr const char* TAG = "export.lf";

bool split_desc(const std::string& s, std::string* nic, std::string* addr) {
    auto pos = s.find('|');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
    *nic = s.substr(0, pos);
    *addr = s.substr(pos + 1);
    return true;
}
}  // namespace

// Referenced (with C++ linkage) from export_backend.cpp's send_libfabric decl.
ExportResult send_libfabric(const ExportRequest& req, const void* data,
                            uint64_t nbytes) {
    ExportResult r;
    r.transport = 3;  // TRANSPORT_LIBFABRIC_SEND
    std::string nic, addr_hex;
    if (!split_desc(req.target_descriptor, &nic, &addr_hex)) {
        r.message = "bad libfabric target (want <nic>|<ib_address_hex>)";
        return r;
    }
    try {
        // north_comm opens a local network on our own NIC and connects to the
        // receiver's advertised IB address. The descriptor's nic is the
        // RECEIVER's device name; the daemon's local device may differ, so allow
        // an explicit override via OFLD_LIBFABRIC_NIC (falls back to the
        // descriptor nic when unset, which is correct on a homogeneous fabric).
        const char* env_nic = std::getenv("OFLD_LIBFABRIC_NIC");
        std::string local_nic = (env_nic && env_nic[0]) ? env_nic : nic;
        auto network = north_comm::GetNetwork(local_nic);
        auto node = std::make_shared<north_comm::LibfabricEndpointNode>(
            north_comm::IbAddress(addr_hex), network);
        north_comm::LibfabricEndpoint ep(node);
        ep.Connect();

        // Stage into a north_comm host Tensor buffer (1-D uint8, nbytes long)
        // so the receiver can recover it losslessly via DLPack. A Bytes buffer
        // would force a UTF-8 str round-trip that corrupts binary weights.
        std::array<int64_t, north_comm::kMaxTensorDims> shape{};
        std::array<int64_t, north_comm::kMaxTensorDims> strides{};
        shape[0] = static_cast<int64_t>(nbytes);
        strides[0] = 1;
        DLDataType u8{kDLUInt, 8, 1};
        north_comm::Tensor::Metadata meta(north_comm::Device(kDLCPU, 0),
                                          /*ndim=*/1, u8, shape, strides,
                                          /*byte_offset=*/0, /*is_contiguous=*/true);
        north_comm::Buffer buf = north_comm::Buffer::Alloc<north_comm::Tensor>(meta);
        std::memcpy(buf->data, data, nbytes);
        network->RegisterSendMemory(buf);
        ep.Send(buf);
        network->UnregisterMemory(buf);
        ep.Close();

        r.ok = true;
        r.bytes_sent = nbytes;
        r.transport_metadata = "libfabric:" + nic;
        r.message = "ok";
    } catch (const std::exception& e) {
        r.message = std::string("libfabric send failed: ") + e.what();
        OFLD_WARN(TAG, "%s", r.message.c_str());
    }
    return r;
}

}  // namespace offload
