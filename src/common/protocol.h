// Control-plane wire protocol for the centralized offload runtime.
//
// This is the hand-rolled, fixed-layout binary encoding that mirrors
// rpc/offload_daemon.proto field-for-field. We deliberately do NOT link a
// protobuf runtime, because libtorch statically bundles its own protobuf and
// the rank agent is loaded into the torch process; linking a second protobuf
// would risk ODR/symbol clashes. We own both ends of this socket, the schema
// is fixed, so a compact little-endian codec is both safe and faster.
//
// Framing (see wire.h): [u32 magic][u16 opcode][u16 flags][u32 payload_len][payload...]
// Out-of-band file descriptors travel via SCM_RIGHTS on RegisterRank only.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "offload_abi.h"

namespace offload {

// Wire framing magic: 'OFRP' (Offload RPC).
static constexpr uint32_t OFLD_RPC_MAGIC = 0x4F465250u;
static constexpr uint32_t OFLD_RPC_WIRE_VERSION = 1u;

// Opcodes. Requests are even-facing; responses reuse the same opcode on the
// reply frame so a client can assert the reply matches the request it sent.
enum class OpCode : uint16_t {
    kInvalid          = 0,
    kRegisterRank     = 1,
    kHeartbeat        = 2,
    kRequestOffload   = 3,
    kMarkD2HSubmitted = 4,
    kMarkD2HComplete  = 5,
    kRequestPrefetch  = 6,
    kMarkH2DSubmitted = 7,
    kMarkH2DComplete  = 8,
    kReleaseLease     = 9,
    kQueryLocation    = 10,
    kBatchComplete    = 11,
    kShutdown         = 12,   // administrative: ask daemon to exit cleanly
    kGetStats         = 13,   // administrative: fetch daemon-wide metrics snapshot

    // ---- v2 canonical model-state extension. Additive: the v1 handlers above
    // are unchanged. Message bodies + codecs live in protocol_v2.h / wire_v2.*.
    // See rpc/offload_canonical.proto and docs/ARCHITECTURE.md. ----
    kRegisterJob            = 20,
    kRequestCanonicalEvict  = 21,
    kCommitCanonicalObject  = 22,
    kSealModelVersion       = 23,
    kGetLatestSealedVersion = 24,
    kGetManifest            = 25,
    kPullTensor             = 26,
    // Local read-back of a canonical object by object_id (trainer offload/reload
    // round-trip). Strictly read-only: never frees/migrates the object backing.
    kRequestCanonicalRestore = 27,
    kReleaseCanonicalRestore = 28,
};

const char* op_name(OpCode op);

// Event type constants shared across completion messages / trace.
enum class EventType : uint32_t {
    kNone         = 0,
    kD2HComplete  = 1,
    kH2DComplete  = 2,
    kReadbackDone = 3,
    kDrainDone    = 4,
    kError        = 5,
};

// ---------------------------------------------------------------------------
// Message structs. Plain data; encode/decode live in wire.cpp.
// ---------------------------------------------------------------------------

struct ArenaFd {
    uint64_t arena_id = 0;
    uint32_t numa_node = 0;
    uint32_t kind = 0;                 // ofld_arena_kind_t
    uint64_t size = 0;
    uint64_t base_offset = 0;
    uint32_t preferred_gpu = 0;
    uint64_t flags = 0;
    uint32_t fd_index = 0;             // index into the SCM_RIGHTS fd array
    uint32_t registration_granularity = 0;
    uint32_t allocation_granularity = 0;
};

struct RegisterRankRequest {
    uint32_t rank_id = 0;
    uint32_t local_rank = 0;
    uint32_t world_rank = 0;
    uint32_t gpu_id = 0;
    uint32_t numa_node = 0;
    uint64_t pid = 0;
    uint64_t capabilities = 0;
};

struct RegisterRankResponse {
    bool ok = false;
    std::string message;
    uint64_t rank_epoch = 0;
    uint64_t control_generation = 0;
    std::vector<ArenaFd> arenas;       // fds transported out-of-band, ordered by fd_index
    uint32_t control_fd_index = 0;
    uint64_t control_size = 0;         // size of the control/slot-table mapping
    uint32_t num_slots = 0;
    // Completion-ring index assigned to this session in [0, max_ranks), or
    // 0xFFFFFFFF if rings are disabled/unavailable (rank falls back to RPC).
    uint32_t ring_index = 0xFFFFFFFFu;
};

struct HeartbeatRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t timestamp_ns = 0;
};

struct HeartbeatResponse {
    bool ok = false;
    std::string message;
};

struct RequestOffloadRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint64_t nbytes = 0;
    uint32_t gpu_id = 0;
    uint32_t numa_node = 0;
    uint32_t alignment = 0;
    uint32_t priority = 0;
    uint64_t flags = 0;
    std::string debug_name;
};

struct RequestOffloadResponse {
    bool ok = false;
    std::string message;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;
    uint64_t arena_id = 0;
    uint64_t arena_offset = 0;
    uint64_t capacity = 0;
    uint32_t state = 0;                // ofld_slot_state_t
    bool blocked = false;              // reservation refused due to pinned pressure
};

struct MarkD2HSubmittedRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t lease_id = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint32_t slot_id = 0;
    uint64_t submit_seq = 0;
    uint64_t timestamp_ns = 0;
};

struct MarkD2HSubmittedResponse {
    bool ok = false;
    std::string message;
};

struct MarkD2HCompleteRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t lease_id = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint32_t slot_id = 0;
    uint64_t complete_seq = 0;
    uint64_t timestamp_ns = 0;
};

struct MarkD2HCompleteResponse {
    bool ok = false;
    std::string message;
    bool latest_version = false;
    bool drain_enqueued = false;
};

struct RequestPrefetchRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;              // 0 => latest known version
    uint64_t nbytes = 0;
    uint32_t gpu_id = 0;
    uint32_t numa_node = 0;
    uint32_t priority = 0;
    uint64_t flags = 0;
};

struct RequestPrefetchResponse {
    bool ok = false;
    std::string message;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;
    uint64_t arena_id = 0;
    uint64_t arena_offset = 0;
    uint64_t nbytes = 0;
    uint32_t state = 0;
    bool ready = false;                // true => PINNED_VALID, safe to H2D now
    uint64_t version = 0;             // resolved version being restored
};

struct MarkH2DSubmittedRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t lease_id = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint32_t slot_id = 0;
    uint64_t submit_seq = 0;
    uint64_t timestamp_ns = 0;
};

struct MarkH2DSubmittedResponse {
    bool ok = false;
    std::string message;
};

struct MarkH2DCompleteRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t lease_id = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint32_t slot_id = 0;
    bool keep_pinned = false;
    uint64_t complete_seq = 0;
    uint64_t timestamp_ns = 0;
};

struct MarkH2DCompleteResponse {
    bool ok = false;
    std::string message;
};

struct ReleaseLeaseRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    uint64_t lease_id = 0;
    uint32_t slot_id = 0;
    std::string reason;
};

struct ReleaseLeaseResponse {
    bool ok = false;
    std::string message;
};

struct LocationQueryRequest {
    uint64_t tensor_id = 0;
    uint64_t version = 0;              // 0 => latest
};

struct LocationQueryResponse {
    bool ok = false;
    std::string message;
    uint32_t location_kind = 0;        // ofld_location_kind_t
    uint32_t slot_id = 0;
    uint64_t nbytes = 0;
    uint64_t version = 0;
};

struct BatchCompletion {
    uint32_t event_type = 0;
    uint64_t lease_id = 0;
    uint64_t tensor_id = 0;
    uint64_t version = 0;
    uint32_t slot_id = 0;
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
    bool keep_pinned = false;          // relevant for H2D completions
};

struct BatchCompleteRequest {
    uint32_t rank_id = 0;
    uint64_t rank_epoch = 0;
    std::vector<BatchCompletion> completions;
};

struct BatchCompleteResponse {
    bool ok = false;
    std::string message;
    uint32_t accepted = 0;
    uint32_t rejected = 0;
};

// Administrative snapshot of daemon-wide counters (see metrics.h names).
struct GetStatsRequest {
    uint32_t rank_id = 0;              // 0xFFFFFFFF => all ranks / global
};

struct GetStatsResponse {
    bool ok = false;
    std::string message;
    // Flat key/value list so we don't couple the wire to the metric set.
    std::vector<std::string> keys;
    std::vector<uint64_t> values;
};

struct ShutdownRequest {
    uint64_t token = 0;                // reserved for future auth; 0 accepted
};

struct ShutdownResponse {
    bool ok = false;
    std::string message;
};

}  // namespace offload
