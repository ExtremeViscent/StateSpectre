// Byte-level codecs for the v2 canonical model-state messages (protocol_v2.h).
//
// Same little-endian, length-prefixed encoding as wire.h — we reuse the Writer/
// Reader primitives and the make_frame()/parse_frame_header() framing verbatim.
// This header only adds encode()/decode_*() for the v2 message bodies.

#pragma once

#include <cstdint>
#include <vector>

#include "protocol_v2.h"
#include "wire.h"  // Writer, Reader, WireError, make_frame, frame header

namespace offload {

#define OFLD_CODEC_V2(T)                                     \
    std::vector<uint8_t> encode(const T& m);                 \
    T decode_##T(const uint8_t* data, size_t len);

OFLD_CODEC_V2(RegisterJobRequest)
OFLD_CODEC_V2(RegisterJobResponse)
OFLD_CODEC_V2(RequestCanonicalEvictRequest)
OFLD_CODEC_V2(RequestCanonicalEvictResponse)
OFLD_CODEC_V2(CommitCanonicalObjectRequest)
OFLD_CODEC_V2(CommitCanonicalObjectResponse)
OFLD_CODEC_V2(SealModelVersionRequest)
OFLD_CODEC_V2(SealModelVersionResponse)
OFLD_CODEC_V2(GetLatestSealedVersionRequest)
OFLD_CODEC_V2(GetLatestSealedVersionResponse)
OFLD_CODEC_V2(GetManifestRequest)
OFLD_CODEC_V2(GetManifestResponse)
OFLD_CODEC_V2(PullTensorRequest)
OFLD_CODEC_V2(PullTensorResponse)
OFLD_CODEC_V2(RequestCanonicalRestoreRequest)
OFLD_CODEC_V2(RequestCanonicalRestoreResponse)
OFLD_CODEC_V2(ReleaseCanonicalRestoreRequest)
OFLD_CODEC_V2(ReleaseCanonicalRestoreResponse)
OFLD_CODEC_V2(DropCanonicalVersionRequest)
OFLD_CODEC_V2(DropCanonicalVersionResponse)
OFLD_CODEC_V2(ReleaseCanonicalRequest)
OFLD_CODEC_V2(ReleaseCanonicalResponse)

#undef OFLD_CODEC_V2

}  // namespace offload
