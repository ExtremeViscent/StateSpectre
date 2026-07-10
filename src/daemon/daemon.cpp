// OffloadDaemon implementation. See daemon.h and spec/*.md.
//
// CUDA-FREE. Owns fd-backed pinned arenas + control shm, slot allocator,
// leases, location table, cold store, drain/readback workers, heartbeat
// monitor. Mutex discipline: mu_ guards all tables; long IO/memcpy in workers
// runs WITHOUT mu_ (copy metadata out, unlock, do IO, relock to commit).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "daemon.h"

#include <fcntl.h>
#include <liburing.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <stdexcept>

#include "metrics.h"
#include "numa_util.h"
#include "uds.h"
#include "util.h"
#include "wire.h"

namespace offload {

namespace {
constexpr const char* TAG = "daemon";

uint64_t roundup(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// memfd_create wrapper. glibc >= 2.27 provides the ::memfd_create symbol when
// _GNU_SOURCE is defined (see top of file); fall back to the raw syscall if the
// libc wrapper is somehow unavailable.
int make_memfd(const char* name, unsigned flags) {
#if defined(MFD_CLOEXEC)
    return ::memfd_create(name, flags);
#elif defined(SYS_memfd_create)
    return static_cast<int>(::syscall(SYS_memfd_create, name, flags));
#else
    errno = ENOSYS;
    return -1;
#endif
}
}  // namespace

// ============================================================================
// Construction / setup
// ============================================================================
OffloadDaemon::OffloadDaemon(const DaemonConfig& cfg) : cfg_(cfg) {
    daemon_epoch_ = now_real_ns();
    // Stripe count: explicit config wins; else 4 when striping, 1 otherwise.
    if (cfg_.nvme_stripe_count > 0)
        nvme_stripe_count_ = cfg_.nvme_stripe ? cfg_.nvme_stripe_count : 1u;
    else
        nvme_stripe_count_ = cfg_.nvme_stripe ? 4u : 1u;
    // Initialize per-stripe append cursors.
    nvme_stripe_offsets_ = std::vector<std::atomic<uint64_t>>(nvme_stripe_count_);
    for (auto& o : nvme_stripe_offsets_) o.store(0);
    build_arenas_and_control();
    build_slot_plan();
}

OffloadDaemon::~OffloadDaemon() { teardown(); }

void OffloadDaemon::build_arenas_and_control() {
    // ---- data arenas: one memfd per NUMA node ----
    arenas_.reserve(cfg_.per_numa.size());
    uint32_t total_slots = 0;
    for (size_t i = 0; i < cfg_.per_numa.size(); ++i) {
        const NumaArenaConfig& ac = cfg_.per_numa[i];
        Arena a;
        a.arena_id = i;
        a.numa_node = ac.numa_node;
        a.size = ac.size_bytes;
        a.registration_granularity =
            static_cast<uint32_t>(std::min<uint64_t>(cfg_.registration_chunk_bytes,
                                                     0xFFFFFFFFull));
        a.allocation_granularity =
            static_cast<uint32_t>(cfg_.allocation_granularity_bytes);

        char name[64];
        std::snprintf(name, sizeof(name), "ofld_arena_%zu", i);
        a.fd = make_memfd(name, MFD_CLOEXEC);
        if (a.fd < 0)
            throw std::runtime_error(std::string("memfd_create(arena) failed: ") +
                                     std::strerror(errno));
        if (::ftruncate(a.fd, static_cast<off_t>(a.size)) != 0)
            throw std::runtime_error(std::string("ftruncate(arena) failed: ") +
                                     std::strerror(errno));
        a.base = ::mmap(nullptr, a.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        a.fd, 0);
        if (a.base == MAP_FAILED)
            throw std::runtime_error(std::string("mmap(arena) failed: ") +
                                     std::strerror(errno));

        // NUMA localization: first-touch + bind from a thread pinned to node.
        int node = static_cast<int>(ac.numa_node);
        void* base = a.base;
        uint64_t sz = a.size;
        std::thread t([node, base, sz]() {
            pin_thread_to_node(node);
            int rc = bind_range_to_node(base, sz, node, /*move_existing=*/false);
            if (rc != 0) {
                OFLD_WARN(TAG, "bind_range_to_node(node=%d) failed rc=%d (non-fatal)",
                          node, rc);
            }
            first_touch_on_node(base, sz, node);
        });
        t.join();

        OFLD_INFO(TAG, "arena %zu: node=%u size=%llu bytes fd=%d", i, ac.numa_node,
                  (unsigned long long)a.size, a.fd);

        // count slots for this arena
        uint64_t nslots = a.size / cfg_.allocation_granularity_bytes;
        total_slots += static_cast<uint32_t>(nslots);
        arenas_.push_back(a);
    }

    // ---- control region ----
    uint32_t num_arenas = static_cast<uint32_t>(arenas_.size());
    // Reserve one completion ring per potential rank in the control region.
    uint32_t ring_cap = cfg_.completion_ring_capacity;
    // capacity must be a power of two; round up.
    if (ring_cap < 2) ring_cap = 2;
    { uint32_t p = 1; while (p < ring_cap) p <<= 1; ring_cap = p; }
    uint32_t ring_max = cfg_.completion_ring_max_ranks;
    uint32_t ring_stride = static_cast<uint32_t>(completion_ring_bytes(ring_cap));
    uint64_t ring_area = (ring_max == 0) ? 0
                         : static_cast<uint64_t>(ring_max) * ring_stride;
    ring_capacity_ = ring_cap;
    ring_stride_ = ring_stride;
    ring_max_ranks_ = ring_max;

    layout_ = compute_control_layout(num_arenas, total_slots, ring_area);
    layout_.max_ranks = ring_max;
    layout_.ring_stride = ring_stride;
    control_fd_ = make_memfd("ofld_control", MFD_CLOEXEC);
    if (control_fd_ < 0)
        throw std::runtime_error(std::string("memfd_create(control) failed: ") +
                                 std::strerror(errno));
    if (::ftruncate(control_fd_, static_cast<off_t>(layout_.total_bytes)) != 0)
        throw std::runtime_error(std::string("ftruncate(control) failed: ") +
                                 std::strerror(errno));
    control_base_ = ::mmap(nullptr, layout_.total_bytes, PROT_READ | PROT_WRITE,
                           MAP_SHARED, control_fd_, 0);
    if (control_base_ == MAP_FAILED)
        throw std::runtime_error(std::string("mmap(control) failed: ") +
                                 std::strerror(errno));
    std::memset(control_base_, 0, layout_.total_bytes);

    header_ = control_header(control_base_);
    header_->magic = OFLD_MAGIC;
    header_->abi_version = OFLD_ABI_VERSION;
    header_->num_slots = layout_.num_slots;
    header_->num_arenas = num_arenas;
    header_->generation = 1;
    header_->daemon_pid = static_cast<uint64_t>(::getpid());
    header_->daemon_epoch = daemon_epoch_;
    header_->heartbeat_ns.store(now_real_ns(), std::memory_order_release);
    header_->flags = 0;
    header_->arena_desc_offset = layout_.arena_desc_offset;
    header_->slot_entries_offset = layout_.slot_entries_offset;
    header_->control_ring_offset = layout_.control_ring_offset;

    arena_descs_ = control_arena_descs(control_base_, header_);
    for (uint32_t i = 0; i < num_arenas; ++i) {
        ofld_arena_desc_t& d = arena_descs_[i];
        d.arena_id = arenas_[i].arena_id;
        d.base_offset = 0;
        d.size = arenas_[i].size;
        d.numa_node = arenas_[i].numa_node;
        d.kind = OFLD_ARENA_DATA_PINNED;
        d.registration_granularity = arenas_[i].registration_granularity;
        d.allocation_granularity = arenas_[i].allocation_granularity;
        d.preferred_gpu = 0;
        d.flags = 0;
    }

    slots_ = control_slots(control_base_, header_);

    // ---- completion rings ----
    // Publish ring-area metadata into the header's reserved bytes and
    // initialize each ring header (capacity, empty indices, unassigned owner).
    RingAreaMeta* rm = ring_area_meta(header_);
    rm->max_ranks = ring_max_ranks_;
    rm->ring_stride = ring_stride_;
    rm->ring_capacity = ring_capacity_;
    rm->_pad = 0;
    for (uint32_t i = 0; i < ring_max_ranks_; ++i) {
        CompletionRingHeader* rh = ring_for_index(control_base_, header_, i);
        rh->capacity = ring_capacity_;
        rh->owner_rank = 0xFFFFFFFFu;
        rh->rank_epoch = 0;
        rh->head.store(0, std::memory_order_relaxed);
        rh->tail.store(0, std::memory_order_relaxed);
    }

    OFLD_INFO(TAG, "control region: %u slots, %u arenas, %u rings(cap=%u), %llu bytes",
              layout_.num_slots, num_arenas, ring_max_ranks_, ring_capacity_,
              (unsigned long long)layout_.total_bytes);
}

void OffloadDaemon::build_slot_plan() {
    const uint64_t gran = cfg_.allocation_granularity_bytes;
    uint32_t global_slot = 0;

    for (size_t ai = 0; ai < arenas_.size(); ++ai) {
        const NumaArenaConfig& ac = cfg_.per_numa[ai];
        const Arena& arena = arenas_[ai];
        uint64_t cursor = 0;  // byte offset within arena

        auto carve_window = [&](uint32_t gpu, bool is_overflow, uint64_t bytes) {
            uint64_t nslots = bytes / gran;
            if (nslots == 0) return;
            Window w;
            w.numa_node = ac.numa_node;
            w.gpu = is_overflow ? kOverflowGpu : gpu;
            w.is_overflow = is_overflow;
            w.slot_ids.reserve(nslots);
            for (uint64_t s = 0; s < nslots; ++s) {
                uint32_t id = global_slot++;
                ofld_slot_entry_t& e = slots_[id];
                e.slot_id = id;
                e.numa_node = ac.numa_node;
                e.gpu_preferred = w.gpu;
                e.arena_id = arena.arena_id;
                e.arena_offset = cursor + s * gran;
                e.capacity = gran;
                e.nbytes = 0;
                e.tensor_id = 0;
                e.version = 0;
                e.lease_id = 0;
                e.owner_rank = 0;
                e.owner_pid = 0;
                e.rank_epoch = 0;
                e.flags = is_overflow ? 0 : OFLD_FLAG_BORROWABLE;
                e.cold_ref = 0;
                e.checksum = 0;
                e.last_touch_ns = 0;
                e.submit_seq = 0;
                e.complete_seq = 0;
                e.state.store(OFLD_SLOT_FREE, std::memory_order_release);
                w.slot_ids.push_back(id);
            }
            cursor += nslots * gran;
            uint32_t idx = static_cast<uint32_t>(windows_.size());
            window_index_[window_key(ac.numa_node, w.gpu)] = idx;
            windows_.push_back(std::move(w));
        };

        for (const auto& gw : ac.gpu_windows) carve_window(gw.gpu, false, gw.size_bytes);
        carve_window(kOverflowGpu, true, ac.overflow_bytes);
    }

    OFLD_INFO(TAG, "slot plan: %u windows, %u slots total",
              static_cast<uint32_t>(windows_.size()), global_slot);
    if (global_slot != layout_.num_slots) {
        OFLD_WARN(TAG, "carved %u slots but layout reserved %u (arena rounding)",
                  global_slot, layout_.num_slots);
    }
}

void OffloadDaemon::teardown() {
    if (control_base_ && control_base_ != MAP_FAILED) {
        ::munmap(control_base_, layout_.total_bytes);
        control_base_ = nullptr;
    }
    if (control_fd_ >= 0) { ::close(control_fd_); control_fd_ = -1; }
    for (auto& a : arenas_) {
        if (a.base && a.base != MAP_FAILED) ::munmap(a.base, a.size);
        if (a.fd >= 0) ::close(a.fd);
        a.base = nullptr;
        a.fd = -1;
    }
    // free any pageable cold buffers still resident
    for (auto& kv : cold_store_) {
        if (kv.second.pageable) { std::free(kv.second.pageable); kv.second.pageable = nullptr; }
    }
}

// ============================================================================
// Slot / control helpers
// ============================================================================
void OffloadDaemon::set_slot_state(uint32_t id, ofld_slot_state_t st) {
    slots_[id].last_touch_ns = now_real_ns();
    slots_[id].state.store(static_cast<uint32_t>(st), std::memory_order_release);
}
uint32_t OffloadDaemon::slot_state(uint32_t id) const {
    return slots_[id].state.load(std::memory_order_acquire);
}
void* OffloadDaemon::slot_addr(uint32_t id) {
    const ofld_slot_entry_t& e = slots_[id];
    Arena& a = arenas_[e.arena_id];
    return static_cast<char*>(a.base) + e.arena_offset;
}
uint32_t OffloadDaemon::slots_for_bytes(uint64_t nbytes) const {
    uint64_t gran = cfg_.allocation_granularity_bytes;
    uint64_t n = (nbytes + gran - 1) / gran;
    if (n == 0) n = 1;
    return static_cast<uint32_t>(n);
}
uint64_t OffloadDaemon::used_pinned_bytes() const {
    return Metrics::instance().get(Metric::kUsedPinnedBytes);
}
double OffloadDaemon::pinned_pressure() const {
    uint64_t total = static_cast<uint64_t>(layout_.num_slots) *
                     cfg_.allocation_granularity_bytes;
    if (total == 0) return 0.0;
    return static_cast<double>(used_pinned_bytes()) / static_cast<double>(total);
}

// Find `count` consecutive FREE slots in window w. Returns base global slot id
// or UINT32_MAX if none. mu_ held.
static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;

// ============================================================================
// Session / lease / location helpers
// ============================================================================
bool OffloadDaemon::epoch_valid(uint32_t rank_id, uint64_t rank_epoch) {
    auto it = sessions_.find(rank_id);
    if (it == sessions_.end() || !it->second.alive) return false;
    if (it->second.rank_epoch != rank_epoch) return false;
    return true;
}

OffloadDaemon::Lease* OffloadDaemon::find_lease(uint64_t lease_id) {
    auto it = leases_.find(lease_id);
    return it == leases_.end() ? nullptr : &it->second;
}

uint64_t OffloadDaemon::latest_version(uint64_t tensor_id) const {
    auto it = locations_.find(tensor_id);
    return it == locations_.end() ? 0 : it->second.version;
}

bool OffloadDaemon::update_location_if_latest(uint64_t tensor_id, uint64_t version,
                                              ofld_location_kind_t kind,
                                              uint32_t slot_id, uint64_t cold_ref,
                                              uint64_t nbytes) {
    auto it = locations_.find(tensor_id);
    if (it != locations_.end() && version < it->second.version) {
        // Stale completion: record it but do not overwrite latest.
        Metrics::instance().inc(Metric::kStaleVersionRejected);
        return false;
    }
    Location loc;
    loc.tensor_id = tensor_id;
    loc.version = version;
    loc.kind = kind;
    loc.slot_id = slot_id;
    loc.cold_ref = cold_ref;
    loc.nbytes = nbytes;
    locations_[tensor_id] = loc;
    return true;
}

bool OffloadDaemon::should_drain(const ofld_slot_entry_t* s) const {
    (void)s;
    return cfg_.drain_on_d2h_complete;
}

// ============================================================================
// Allocator
// ============================================================================
// Reserve `count` contiguous FREE slots in a window; on success mark them with
// the given metadata + initial_state and return the base slot id. mu_ held.
bool OffloadDaemon::try_alloc_in_window(const Window& w, uint32_t count,
                                        uint32_t* base_out) {
    const auto& ids = w.slot_ids;
    if (ids.size() < count) return false;
    for (size_t i = 0; i + count <= ids.size(); ++i) {
        // ids are contiguous within one arena in address order.
        bool run_ok = true;
        for (uint32_t k = 0; k < count; ++k) {
            if (slot_state(ids[i + k]) != OFLD_SLOT_FREE) { run_ok = false; i += k; break; }
        }
        if (run_ok) {
            *base_out = ids[i];
            return true;
        }
    }
    return false;
}

bool OffloadDaemon::allocate_slots(uint32_t gpu_id, uint32_t numa_node,
                                   uint64_t nbytes, uint64_t flags,
                                   bool for_prefetch, uint32_t* base_out,
                                   uint32_t* count_out, bool* blocked_out) {
    *blocked_out = false;
    uint32_t count = slots_for_bytes(nbytes);
    *count_out = count;
    uint32_t base = kNoSlot;
    bool remote = false;

    auto try_win = [&](uint32_t numa, uint32_t gpu) -> bool {
        auto it = window_index_.find(window_key(numa, gpu));
        if (it == window_index_.end()) return false;
        return try_alloc_in_window(windows_[it->second], count, &base);
    };

    // 1. current GPU preferred window (same NUMA)
    if (try_win(numa_node, gpu_id)) { /* base set */ }
    // 2. same-NUMA overflow window (default allow; ALLOW_OVERFLOW is a no-op hint)
    else if (try_win(numa_node, kOverflowGpu)) { /* base set */ }
    // 3. borrowable same-NUMA peer GPU windows
    else {
        for (const auto& w : windows_) {
            if (w.numa_node != numa_node || w.is_overflow) continue;
            if (w.gpu == gpu_id) continue;
            if (try_alloc_in_window(w, count, &base)) break;
        }
        // 5. remote-NUMA fallback (only if allowed)
        if (base == kNoSlot && (flags & OFLD_FLAG_ALLOW_REMOTE_NUMA)) {
            for (const auto& w : windows_) {
                if (w.numa_node == numa_node) continue;
                if (try_alloc_in_window(w, count, &base)) { remote = true; break; }
            }
        }
    }

    if (base == kNoSlot) return false;

    if (remote) Metrics::instance().inc(Metric::kRemoteNumaAllocs);
    *base_out = base;
    return true;
}

void OffloadDaemon::free_slots(uint32_t base_slot, uint32_t slot_count) {
    for (uint32_t k = 0; k < slot_count; ++k) {
        uint32_t id = base_slot + k;
        ofld_slot_entry_t& e = slots_[id];
        e.nbytes = 0;
        e.tensor_id = 0;
        e.version = 0;
        e.lease_id = 0;
        e.owner_rank = 0;
        e.owner_pid = 0;
        e.rank_epoch = 0;
        e.cold_ref = 0;
        e.submit_seq = 0;
        e.complete_seq = 0;
        set_slot_state(id, OFLD_SLOT_FREE);
    }
    uint64_t freed = static_cast<uint64_t>(slot_count) *
                     cfg_.allocation_granularity_bytes;
    Metrics::instance().sub(Metric::kUsedPinnedBytes, freed);
}

// ============================================================================
// Connection handling
// ============================================================================
void OffloadDaemon::run() {
    if (::pipe(stop_pipe_) != 0)
        throw std::runtime_error(std::string("pipe() failed: ") + std::strerror(errno));

    listen_fd_ = uds_listen(cfg_.socket_path);
    running_.store(true);
    workers_running_.store(true);

    if (cfg_.nvme_enabled) ensure_nvme_dir();

    // Start workers.
    uint32_t nd = std::max<uint32_t>(1, cfg_.drain_workers);
    for (uint32_t i = 0; i < nd; ++i)
        drain_threads_.emplace_back([this] { drain_worker_loop(); });
    uint32_t nr = std::max<uint32_t>(1, cfg_.drain_workers);
    for (uint32_t i = 0; i < nr; ++i)
        readback_threads_.emplace_back([this] { readback_worker_loop(); });
    heartbeat_thread_ = std::thread([this] { heartbeat_monitor_loop(); });
    if (ring_max_ranks_ > 0)
        ring_poller_thread_ = std::thread([this] { ring_poller_loop(); });

    OFLD_INFO(TAG, "daemon running (pid=%d) socket=%s", (int)::getpid(),
              cfg_.socket_path.c_str());

    accept_loop();

    // Shutting down: stop workers, join everything.
    workers_running_.store(false);
    drain_cv_.notify_all();
    readback_cv_.notify_all();

    // Unblock and join connection threads.
    {
        std::lock_guard<std::mutex> g(conn_threads_mu_);
        for (int fd : conn_fds_) ::shutdown(fd, SHUT_RDWR);
    }
    for (auto& t : conn_threads_) if (t.joinable()) t.join();
    conn_threads_.clear();

    for (auto& t : drain_threads_) if (t.joinable()) t.join();
    for (auto& t : readback_threads_) if (t.joinable()) t.join();
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    if (ring_poller_thread_.joinable()) ring_poller_thread_.join();
    drain_threads_.clear();
    readback_threads_.clear();

    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
    ::unlink(cfg_.socket_path.c_str());
    if (stop_pipe_[0] >= 0) { ::close(stop_pipe_[0]); stop_pipe_[0] = -1; }
    if (stop_pipe_[1] >= 0) { ::close(stop_pipe_[1]); stop_pipe_[1] = -1; }
    OFLD_INFO(TAG, "daemon stopped");
}

void OffloadDaemon::request_stop() {
    running_.store(false);
    if (stop_pipe_[1] >= 0) {
        const char b = 1;
        ssize_t r = ::write(stop_pipe_[1], &b, 1);
        (void)r;  // best-effort wakeup
    }
}

void OffloadDaemon::accept_loop() {
    while (running_.load()) {
        struct pollfd pfds[2];
        pfds[0].fd = listen_fd_;
        pfds[0].events = POLLIN;
        pfds[0].revents = 0;
        pfds[1].fd = stop_pipe_[0];
        pfds[1].events = POLLIN;
        pfds[1].revents = 0;
        int rc = ::poll(pfds, 2, -1);
        if (rc < 0) {
            if (errno == EINTR) continue;
            OFLD_ERR(TAG, "poll failed: %s", std::strerror(errno));
            break;
        }
        if (pfds[1].revents & POLLIN) break;  // stop requested
        if (pfds[0].revents & POLLIN) {
            int fd = uds_accept(listen_fd_);
            if (fd < 0) continue;
            {
                std::lock_guard<std::mutex> g(conn_threads_mu_);
                conn_fds_.insert(fd);
                conn_threads_.emplace_back([this, fd] { serve_connection(fd); });
            }
        }
    }
}

void OffloadDaemon::serve_connection(int fd) {
    for (;;) {
        OpCode op;
        std::vector<uint8_t> payload;
        std::vector<int> fds;
        bool got;
        try {
            got = recv_frame(fd, &op, &payload, &fds);
        } catch (const std::exception& e) {
            OFLD_DEBUG(TAG, "connection recv error: %s", e.what());
            break;
        }
        for (int f : fds) close_fd(f);  // clients send no fds; close if any
        if (!got) break;                // clean EOF
        try {
            dispatch(fd, op, payload);
        } catch (const std::exception& e) {
            OFLD_WARN(TAG, "dispatch error op=%s: %s", op_name(op), e.what());
            break;
        }
        if (!running_.load()) break;
    }
    ::close(fd);
    {
        std::lock_guard<std::mutex> g(conn_threads_mu_);
        conn_fds_.erase(fd);
    }
}

void OffloadDaemon::dispatch(int fd, OpCode op, const std::vector<uint8_t>& p) {
    const uint8_t* d = p.data();
    size_t n = p.size();
    std::vector<uint8_t> body;
    std::vector<int> out_fds;

    switch (op) {
        case OpCode::kRegisterRank: {
            auto resp = handle_register(decode_RegisterRankRequest(d, n), &out_fds);
            body = encode(resp);
            break;
        }
        case OpCode::kHeartbeat: {
            body = encode(handle_heartbeat(decode_HeartbeatRequest(d, n)));
            break;
        }
        case OpCode::kRequestOffload: {
            body = encode(handle_request_offload(decode_RequestOffloadRequest(d, n)));
            break;
        }
        case OpCode::kMarkD2HSubmitted: {
            body = encode(handle_mark_d2h_submitted(decode_MarkD2HSubmittedRequest(d, n)));
            break;
        }
        case OpCode::kMarkD2HComplete: {
            body = encode(handle_mark_d2h_complete(decode_MarkD2HCompleteRequest(d, n)));
            break;
        }
        case OpCode::kRequestPrefetch: {
            body = encode(handle_request_prefetch(decode_RequestPrefetchRequest(d, n)));
            break;
        }
        case OpCode::kMarkH2DSubmitted: {
            body = encode(handle_mark_h2d_submitted(decode_MarkH2DSubmittedRequest(d, n)));
            break;
        }
        case OpCode::kMarkH2DComplete: {
            body = encode(handle_mark_h2d_complete(decode_MarkH2DCompleteRequest(d, n)));
            break;
        }
        case OpCode::kReleaseLease: {
            body = encode(handle_release_lease(decode_ReleaseLeaseRequest(d, n)));
            break;
        }
        case OpCode::kQueryLocation: {
            body = encode(handle_query_location(decode_LocationQueryRequest(d, n)));
            break;
        }
        case OpCode::kBatchComplete: {
            body = encode(handle_batch_complete(decode_BatchCompleteRequest(d, n)));
            break;
        }
        case OpCode::kGetStats: {
            body = encode(handle_get_stats(decode_GetStatsRequest(d, n)));
            break;
        }
        case OpCode::kShutdown: {
            body = encode(handle_shutdown(decode_ShutdownRequest(d, n)));
            break;
        }
        default:
            OFLD_WARN(TAG, "unknown opcode %u", static_cast<unsigned>(op));
            return;
    }

    send_frame(fd, make_frame(op, body), out_fds);
}

// ============================================================================
// RPC handlers
// ============================================================================
RegisterRankResponse OffloadDaemon::handle_register(const RegisterRankRequest& req,
                                                    std::vector<int>* out_fds) {
    std::lock_guard<std::mutex> lk(mu_);
    RegisterRankResponse resp;

    // Invalidate any prior session with the same rank_id or same pid.
    auto invalidate = [&](Session& s) {
        if (!s.alive) return;
        s.alive = false;
        Metrics::instance().inc(Metric::kRankSessionsInvalidated);
        // Recover leases belonging to the dead session.
        std::vector<uint64_t> to_erase;
        for (auto& kv : leases_) {
            Lease& L = kv.second;
            if (L.owner_rank != s.rank_id || L.rank_epoch != s.rank_epoch) continue;
            uint32_t st = slot_state(L.base_slot);
            if (st == OFLD_SLOT_PINNED_VALID || st == OFLD_SLOT_COLD_VALID ||
                st == OFLD_SLOT_DRAIN_IN_FLIGHT) {
                // Data safe / in progress; keep slot, drop the lease binding.
                to_erase.push_back(L.lease_id);
            } else {
                // In-flight (never completed): reclaim.
                set_slot_state(L.base_slot, OFLD_SLOT_ERROR);
                free_slots(L.base_slot, L.slot_count);
                to_erase.push_back(L.lease_id);
            }
        }
        for (uint64_t id : to_erase) leases_.erase(id);
        // Release the session's completion ring (drain any stragglers first so
        // late pushes from the dead rank are applied before we reset it).
        if (s.ring_index < ring_max_ranks_) {
            CompletionRingHeader* rh =
                ring_for_index(control_base_, header_, s.ring_index);
            if (rh) {
                ofld_completion_entry_t e;
                while (ring_pop(rh, &e)) {
                    apply_ring_completion(s.rank_id, s.rank_epoch, e);
                }
                rh->owner_rank = 0xFFFFFFFFu;
                rh->rank_epoch = 0;
            }
            if (s.ring_index < ring_owner_.size())
                ring_owner_[s.ring_index] = 0xFFFFFFFFu;
        }
    };

    auto it = sessions_.find(req.rank_id);
    if (it != sessions_.end()) invalidate(it->second);
    for (auto& kv : sessions_) {
        if (kv.first != req.rank_id && kv.second.alive && kv.second.pid == req.pid)
            invalidate(kv.second);
    }

    uint64_t epoch = next_rank_epoch_++;
    Session s;
    s.rank_id = req.rank_id;
    s.rank_epoch = epoch;
    s.pid = req.pid;
    s.gpu_id = req.gpu_id;
    s.numa_node = req.numa_node;
    s.last_heartbeat_ns = now_real_ns();
    s.alive = true;

    // Assign a completion ring index (first free). If none available, the rank
    // falls back to batched RPC completions (ring_index = 0xFFFFFFFF).
    s.ring_index = 0xFFFFFFFFu;
    if (ring_owner_.size() < ring_max_ranks_) ring_owner_.resize(ring_max_ranks_, 0xFFFFFFFFu);
    for (uint32_t i = 0; i < ring_max_ranks_; ++i) {
        if (ring_owner_[i] == 0xFFFFFFFFu) {
            s.ring_index = i;
            ring_owner_[i] = req.rank_id;
            CompletionRingHeader* rh = ring_for_index(control_base_, header_, i);
            if (rh) {
                // Reset the ring for the new owner before publishing ownership.
                rh->head.store(0, std::memory_order_relaxed);
                rh->tail.store(0, std::memory_order_relaxed);
                rh->owner_rank = req.rank_id;
                std::atomic_thread_fence(std::memory_order_release);
                rh->rank_epoch = epoch;  // epoch != 0 signals "ring active"
            }
            break;
        }
    }
    sessions_[req.rank_id] = s;

    // Build response: all arena fds, then the control fd.
    resp.ok = true;
    resp.message = "ok";
    resp.rank_epoch = epoch;
    resp.control_generation = header_->generation;
    resp.arenas.reserve(arenas_.size());
    out_fds->clear();
    for (size_t i = 0; i < arenas_.size(); ++i) {
        ArenaFd a;
        a.arena_id = arenas_[i].arena_id;
        a.numa_node = arenas_[i].numa_node;
        a.kind = OFLD_ARENA_DATA_PINNED;
        a.size = arenas_[i].size;
        a.base_offset = 0;
        a.preferred_gpu = 0;
        a.flags = 0;
        a.fd_index = static_cast<uint32_t>(i);
        a.registration_granularity = arenas_[i].registration_granularity;
        a.allocation_granularity = arenas_[i].allocation_granularity;
        resp.arenas.push_back(a);
        out_fds->push_back(arenas_[i].fd);
    }
    resp.control_fd_index = static_cast<uint32_t>(arenas_.size());
    out_fds->push_back(control_fd_);
    resp.control_size = layout_.total_bytes;
    resp.num_slots = layout_.num_slots;
    resp.ring_index = s.ring_index;

    OFLD_INFO(TAG, "RegisterRank rank=%u pid=%llu gpu=%u numa=%u -> epoch=%llu ring=%d",
              req.rank_id, (unsigned long long)req.pid, req.gpu_id, req.numa_node,
              (unsigned long long)epoch,
              (s.ring_index == 0xFFFFFFFFu ? -1 : (int)s.ring_index));
    return resp;
}

HeartbeatResponse OffloadDaemon::handle_heartbeat(const HeartbeatRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    HeartbeatResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false;
        resp.message = "epoch invalid";
        return resp;
    }
    sessions_[req.rank_id].last_heartbeat_ns = now_real_ns();
    header_->heartbeat_ns.store(now_real_ns(), std::memory_order_release);
    resp.ok = true;
    resp.message = "ok";
    return resp;
}

RequestOffloadResponse OffloadDaemon::handle_request_offload(
    const RequestOffloadRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    RequestOffloadResponse resp;

    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false;
        resp.message = "epoch invalid";
        Trace::instance().event("ERROR", req.rank_id, req.gpu_id, req.tensor_id,
                                req.version, 0, 0, req.nbytes);
        return resp;
    }
    if (req.nbytes == 0) {
        resp.ok = false;
        resp.message = "nbytes must be > 0";
        return resp;
    }

    Metrics::instance().inc(Metric::kOffloadRequests);
    Trace::instance().event("REQUEST_OFFLOAD", req.rank_id, req.gpu_id,
                            req.tensor_id, req.version, 0, 0, req.nbytes);

    uint32_t count = slots_for_bytes(req.nbytes);
    uint64_t owner_pid = sessions_[req.rank_id].pid;

    // Attempt allocation, forcing synchronous drains under pressure.
    uint32_t base = kNoSlot, ncount = 0;
    bool blocked = false;
    bool got = false;
    const int kMaxDrainAttempts = 64;
    for (int attempt = 0; attempt <= kMaxDrainAttempts; ++attempt) {
        if (allocate_slots(req.gpu_id, req.numa_node, req.nbytes, req.flags,
                           /*for_prefetch=*/false, &base, &ncount, &blocked)) {
            got = true;
            break;
        }
        // Step 4: try to free space by draining an eligible PINNED_VALID run.
        if (!force_drain_one(lk)) break;  // nothing left to drain
    }

    if (!got) {
        Metrics::instance().inc(Metric::kBlockedReservations);
        resp.ok = false;
        resp.blocked = true;
        resp.message = "no pinned capacity (blocked / backpressure)";
        Trace::instance().event("ERROR", req.rank_id, req.gpu_id, req.tensor_id,
                                req.version, 0, 0, req.nbytes);
        return resp;
    }

    // Overlap guard: every slot in the run must have been FREE.
    for (uint32_t k = 0; k < count; ++k) {
        if (slot_state(base + k) != OFLD_SLOT_FREE) {
            Metrics::instance().inc(Metric::kSlotOverlapPrevented);
            resp.ok = false;
            resp.message = "slot overlap prevented";
            return resp;
        }
    }

    uint64_t lease_id = next_lease_id_++;
    // Commit reservation: mark run RESERVED_D2H + metadata.
    for (uint32_t k = 0; k < count; ++k) {
        uint32_t id = base + k;
        ofld_slot_entry_t& e = slots_[id];
        e.tensor_id = req.tensor_id;
        e.version = req.version;
        e.lease_id = lease_id;
        e.owner_rank = req.rank_id;
        e.owner_pid = owner_pid;
        e.rank_epoch = req.rank_epoch;
        e.nbytes = (k == 0) ? req.nbytes : e.capacity;
        e.flags = req.flags;
        set_slot_state(id, OFLD_SLOT_RESERVED_D2H);
    }
    Metrics::instance().add(Metric::kUsedPinnedBytes,
                            static_cast<uint64_t>(count) *
                                cfg_.allocation_granularity_bytes);

    Lease L;
    L.lease_id = lease_id;
    L.base_slot = base;
    L.slot_count = count;
    L.tensor_id = req.tensor_id;
    L.version = req.version;
    L.owner_rank = req.rank_id;
    L.owner_pid = owner_pid;
    L.rank_epoch = req.rank_epoch;
    L.nbytes = req.nbytes;
    L.is_prefetch = false;
    leases_[lease_id] = L;

    Metrics::instance().inc(Metric::kLeasesGranted);

    resp.ok = true;
    resp.message = "ok";
    resp.lease_id = lease_id;
    resp.slot_id = base;
    resp.arena_id = slots_[base].arena_id;
    resp.arena_offset = slots_[base].arena_offset;
    resp.capacity = static_cast<uint64_t>(count) * cfg_.allocation_granularity_bytes;
    resp.state = OFLD_SLOT_RESERVED_D2H;
    resp.blocked = false;

    Trace::instance().event("LEASE_GRANTED", req.rank_id, req.gpu_id, req.tensor_id,
                            req.version, lease_id, base, req.nbytes);
    return resp;
}

MarkD2HSubmittedResponse OffloadDaemon::handle_mark_d2h_submitted(
    const MarkD2HSubmittedRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    MarkD2HSubmittedResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    Lease* L = find_lease(req.lease_id);
    if (!L || L->owner_rank != req.rank_id || L->base_slot != req.slot_id) {
        Metrics::instance().inc(Metric::kLeaseOwnerMismatchRejected);
        resp.ok = false; resp.message = "lease/owner/slot mismatch"; return resp;
    }
    if (slot_state(L->base_slot) != OFLD_SLOT_RESERVED_D2H) {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
        resp.ok = false; resp.message = "not in RESERVED_D2H"; return resp;
    }
    for (uint32_t k = 0; k < L->slot_count; ++k) {
        slots_[L->base_slot + k].submit_seq = req.submit_seq;
        set_slot_state(L->base_slot + k, OFLD_SLOT_D2H_IN_FLIGHT);
    }
    Metrics::instance().add(Metric::kInflightD2HBytes, L->nbytes);
    Trace::instance().event("D2H_SUBMITTED", req.rank_id, 0, req.tensor_id,
                            req.version, req.lease_id, req.slot_id, L->nbytes);
    resp.ok = true; resp.message = "ok";
    return resp;
}

bool OffloadDaemon::apply_d2h_complete(uint32_t rank_id, uint64_t rank_epoch,
                                       uint64_t lease_id, uint64_t tensor_id,
                                       uint64_t version, uint32_t slot_id,
                                       uint64_t complete_seq, bool* out_latest,
                                       bool* out_drain_enqueued) {
    *out_latest = false;
    *out_drain_enqueued = false;
    if (!epoch_valid(rank_id, rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        return false;
    }
    Lease* L = find_lease(lease_id);
    if (!L || L->owner_rank != rank_id || L->base_slot != slot_id) {
        Metrics::instance().inc(Metric::kLeaseOwnerMismatchRejected);
        return false;
    }
    if (L->tensor_id != tensor_id || L->version != version) {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
        return false;
    }
    if (slot_state(L->base_slot) != OFLD_SLOT_D2H_IN_FLIGHT) {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
        return false;
    }

    for (uint32_t k = 0; k < L->slot_count; ++k) {
        slots_[L->base_slot + k].complete_seq = complete_seq;
        set_slot_state(L->base_slot + k, OFLD_SLOT_PINNED_VALID);
    }
    Metrics::instance().sub(Metric::kInflightD2HBytes, L->nbytes);
    Metrics::instance().add(Metric::kD2HBytes, L->nbytes);

    bool latest = update_location_if_latest(tensor_id, version, OFLD_LOC_PINNED,
                                            L->base_slot, 0, L->nbytes);
    *out_latest = latest;

    Trace::instance().event("D2H_COMPLETE", rank_id, 0, tensor_id, version,
                            lease_id, L->base_slot, L->nbytes);

    if (should_drain(&slots_[L->base_slot])) {
        for (uint32_t k = 0; k < L->slot_count; ++k)
            set_slot_state(L->base_slot + k, OFLD_SLOT_DRAIN_IN_FLIGHT);
        {
            std::lock_guard<std::mutex> g(drain_mu_);
            drain_queue_.push_back(DrainJob{L->base_slot, L->slot_count});
        }
        drain_cv_.notify_one();
        Metrics::instance().inc(Metric::kDrainsEnqueued);
        Metrics::instance().add(Metric::kDrainingBytes, L->nbytes);
        *out_drain_enqueued = true;
        Trace::instance().event("DRAIN_ENQUEUED", rank_id, 0, tensor_id, version,
                                lease_id, L->base_slot, L->nbytes);
    }
    return true;
}

MarkD2HCompleteResponse OffloadDaemon::handle_mark_d2h_complete(
    const MarkD2HCompleteRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    MarkD2HCompleteResponse resp;
    bool latest = false, enq = false;
    bool ok = apply_d2h_complete(req.rank_id, req.rank_epoch, req.lease_id,
                                 req.tensor_id, req.version, req.slot_id,
                                 req.complete_seq, &latest, &enq);
    resp.ok = ok;
    resp.message = ok ? "ok" : "rejected";
    resp.latest_version = latest;
    resp.drain_enqueued = enq;
    if (!ok)
        Trace::instance().event("ERROR", req.rank_id, 0, req.tensor_id, req.version,
                                req.lease_id, req.slot_id, 0);
    return resp;
}

RequestPrefetchResponse OffloadDaemon::handle_request_prefetch(
    const RequestPrefetchRequest& req) {
    std::unique_lock<std::mutex> lk(mu_);
    RequestPrefetchResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }

    Metrics::instance().inc(Metric::kPrefetchRequests);
    auto it = locations_.find(req.tensor_id);
    if (it == locations_.end()) {
        resp.ok = false; resp.message = "unknown tensor"; return resp;
    }
    Location loc = it->second;
    uint64_t want_version = (req.version == 0) ? loc.version : req.version;
    if (want_version != loc.version) {
        resp.ok = false; resp.message = "requested version not latest/available";
        return resp;
    }
    if (loc.kind == OFLD_LOC_NONE || loc.kind == OFLD_LOC_DELETED) {
        resp.ok = false; resp.message = "tensor not resident"; return resp;
    }

    Trace::instance().event("PREFETCH_REQUESTED", req.rank_id, req.gpu_id,
                            req.tensor_id, want_version, 0, loc.slot_id, loc.nbytes);

    // Idempotency: a rank polls RequestPrefetch repeatedly while waiting for a
    // cold->pinned readback. If a readback for this (tensor_id, version) is
    // already in flight (or already landed), return that same slot instead of
    // allocating a second pinned run + enqueuing a duplicate readback. Without
    // this, each poll would leak a fresh readback slot run.
    for (auto& kv : leases_) {
        Lease& EL = kv.second;
        if (!EL.is_prefetch) continue;
        if (EL.tensor_id != req.tensor_id || EL.version != want_version) continue;
        uint32_t st = slot_state(EL.base_slot);
        if (st == OFLD_SLOT_READBACK_IN_FLIGHT || st == OFLD_SLOT_PINNED_VALID) {
            resp.ok = true;
            resp.lease_id = EL.lease_id;
            resp.slot_id = EL.base_slot;
            resp.arena_id = slots_[EL.base_slot].arena_id;
            resp.arena_offset = slots_[EL.base_slot].arena_offset;
            resp.nbytes = EL.nbytes;
            resp.state = st;
            resp.ready = (st == OFLD_SLOT_PINNED_VALID);
            resp.version = want_version;
            resp.message = resp.ready ? "ready" : "readback in flight";
            if (resp.ready)
                Trace::instance().event("PINNED_READY", req.rank_id, req.gpu_id,
                                        req.tensor_id, want_version, EL.lease_id,
                                        EL.base_slot, EL.nbytes);
            return resp;
        }
    }

    uint64_t owner_pid = sessions_[req.rank_id].pid;
    uint64_t lease_id = next_lease_id_++;
    uint32_t count = slots_for_bytes(loc.nbytes);

    if (loc.kind == OFLD_LOC_PINNED && slot_state(loc.slot_id) == OFLD_SLOT_PINNED_VALID) {
        // Already resident and readable. Bind a prefetch lease to the slot.
        Lease L;
        L.lease_id = lease_id;
        L.base_slot = loc.slot_id;
        L.slot_count = count;
        L.tensor_id = req.tensor_id;
        L.version = want_version;
        L.owner_rank = req.rank_id;
        L.owner_pid = owner_pid;
        L.rank_epoch = req.rank_epoch;
        L.nbytes = loc.nbytes;
        L.is_prefetch = true;
        leases_[lease_id] = L;

        resp.ok = true; resp.message = "ok";
        resp.lease_id = lease_id;
        resp.slot_id = loc.slot_id;
        resp.arena_id = slots_[loc.slot_id].arena_id;
        resp.arena_offset = slots_[loc.slot_id].arena_offset;
        resp.nbytes = loc.nbytes;
        resp.state = OFLD_SLOT_PINNED_VALID;
        resp.ready = true;
        resp.version = want_version;
        Trace::instance().event("PINNED_READY", req.rank_id, req.gpu_id,
                                req.tensor_id, want_version, lease_id, loc.slot_id,
                                loc.nbytes);
        return resp;
    }

    // Cold (PAGEABLE / NVME): allocate a pinned run and start readback.
    uint32_t base = kNoSlot, ncount = 0;
    bool blocked = false, got = false;
    for (int attempt = 0; attempt <= 64; ++attempt) {
        if (allocate_slots(req.gpu_id, req.numa_node, loc.nbytes, req.flags,
                           /*for_prefetch=*/true, &base, &ncount, &blocked)) {
            got = true; break;
        }
        if (!force_drain_one(lk)) break;
    }

    if (!got) {
        uint32_t nfree = 0;
        for (uint32_t i = 0; i < layout_.num_slots; ++i)
            if (slot_state(i) == OFLD_SLOT_FREE) ++nfree;
        OFLD_WARN(TAG, "readback alloc failed: need %u contiguous slots for "
                  "%llu bytes, %u/%u slots free (pinned pressure)",
                  slots_for_bytes(loc.nbytes), (unsigned long long)loc.nbytes,
                  nfree, layout_.num_slots);
        Metrics::instance().inc(Metric::kBlockedReservations);
        resp.ok = false; resp.message = "no pinned capacity for readback";
        return resp;
    }

    for (uint32_t k = 0; k < count; ++k) {
        uint32_t id = base + k;
        ofld_slot_entry_t& e = slots_[id];
        e.tensor_id = req.tensor_id;
        e.version = want_version;
        e.lease_id = lease_id;
        e.owner_rank = req.rank_id;
        e.owner_pid = owner_pid;
        e.rank_epoch = req.rank_epoch;
        e.nbytes = (k == 0) ? loc.nbytes : e.capacity;
        e.cold_ref = loc.cold_ref;
        e.flags = req.flags;
        set_slot_state(id, OFLD_SLOT_READBACK_IN_FLIGHT);
    }
    Metrics::instance().add(Metric::kUsedPinnedBytes,
                            static_cast<uint64_t>(count) *
                                cfg_.allocation_granularity_bytes);

    Lease L;
    L.lease_id = lease_id;
    L.base_slot = base;
    L.slot_count = count;
    L.tensor_id = req.tensor_id;
    L.version = want_version;
    L.owner_rank = req.rank_id;
    L.owner_pid = owner_pid;
    L.rank_epoch = req.rank_epoch;
    L.nbytes = loc.nbytes;
    L.is_prefetch = true;
    leases_[lease_id] = L;

    {
        std::lock_guard<std::mutex> g(readback_mu_);
        readback_queue_.push_back(ReadbackJob{base, count, loc.cold_ref, loc.nbytes});
    }
    readback_cv_.notify_one();
    Metrics::instance().inc(Metric::kReadbacksStarted);
    Trace::instance().event("READBACK_STARTED", req.rank_id, req.gpu_id,
                            req.tensor_id, want_version, lease_id, base, loc.nbytes);

    resp.ok = true; resp.message = "readback started";
    resp.lease_id = lease_id;
    resp.slot_id = base;
    resp.arena_id = slots_[base].arena_id;
    resp.arena_offset = slots_[base].arena_offset;
    resp.nbytes = loc.nbytes;
    resp.state = OFLD_SLOT_READBACK_IN_FLIGHT;
    resp.ready = false;
    resp.version = want_version;
    return resp;
}

MarkH2DSubmittedResponse OffloadDaemon::handle_mark_h2d_submitted(
    const MarkH2DSubmittedRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    MarkH2DSubmittedResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    Lease* L = find_lease(req.lease_id);
    if (!L || L->owner_rank != req.rank_id || L->base_slot != req.slot_id) {
        Metrics::instance().inc(Metric::kLeaseOwnerMismatchRejected);
        resp.ok = false; resp.message = "lease/owner/slot mismatch"; return resp;
    }
    if (slot_state(L->base_slot) != OFLD_SLOT_PINNED_VALID) {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
        resp.ok = false; resp.message = "slot not PINNED_VALID"; return resp;
    }
    for (uint32_t k = 0; k < L->slot_count; ++k) {
        slots_[L->base_slot + k].submit_seq = req.submit_seq;
        set_slot_state(L->base_slot + k, OFLD_SLOT_H2D_IN_FLIGHT);
    }
    Metrics::instance().add(Metric::kInflightH2DBytes, L->nbytes);
    Trace::instance().event("H2D_SUBMITTED", req.rank_id, 0, req.tensor_id,
                            req.version, req.lease_id, req.slot_id, L->nbytes);
    resp.ok = true; resp.message = "ok";
    return resp;
}

bool OffloadDaemon::apply_h2d_complete(uint32_t rank_id, uint64_t rank_epoch,
                                       uint64_t lease_id, uint64_t tensor_id,
                                       uint64_t version, uint32_t slot_id,
                                       bool keep_pinned, uint64_t complete_seq) {
    if (!epoch_valid(rank_id, rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        return false;
    }
    Lease* L = find_lease(lease_id);
    if (!L || L->owner_rank != rank_id || L->base_slot != slot_id) {
        Metrics::instance().inc(Metric::kLeaseOwnerMismatchRejected);
        return false;
    }
    if (slot_state(L->base_slot) != OFLD_SLOT_H2D_IN_FLIGHT) {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
        return false;
    }

    uint32_t base = L->base_slot, cnt = L->slot_count;
    uint64_t nbytes = L->nbytes;
    for (uint32_t k = 0; k < cnt; ++k) slots_[base + k].complete_seq = complete_seq;
    Metrics::instance().sub(Metric::kInflightH2DBytes, nbytes);
    Metrics::instance().add(Metric::kH2DBytes, nbytes);
    Trace::instance().event("H2D_COMPLETE", rank_id, 0, tensor_id, version,
                            lease_id, base, nbytes);

    // Decide whether the pinned slot can be released.
    bool cold_exists = false;
    ofld_location_kind_t cold_kind = OFLD_LOC_NONE;
    uint64_t cold_ref = slots_[base].cold_ref;
    auto lit = locations_.find(tensor_id);
    if (lit != locations_.end() && lit->second.cold_ref != 0)
        cold_ref = lit->second.cold_ref;
    if (cold_ref != 0) {
        auto cit = cold_store_.find(cold_ref);
        if (cit != cold_store_.end()) { cold_exists = true; cold_kind = cit->second.kind; }
    }

    if (!keep_pinned && cold_exists) {
        // Release pinned; latest copy remains durable in cold tier.
        if (lit != locations_.end() && lit->second.slot_id == base &&
            lit->second.version == version) {
            lit->second.kind = cold_kind;
            lit->second.slot_id = 0;
        }
        free_slots(base, cnt);
        leases_.erase(lease_id);
    } else {
        // Keep the pinned copy resident and readable.
        for (uint32_t k = 0; k < cnt; ++k)
            set_slot_state(base + k, OFLD_SLOT_PINNED_VALID);
        // Location remains PINNED at this slot for the latest version.
        update_location_if_latest(tensor_id, version, OFLD_LOC_PINNED, base,
                                  cold_ref, nbytes);
    }
    return true;
}

MarkH2DCompleteResponse OffloadDaemon::handle_mark_h2d_complete(
    const MarkH2DCompleteRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    MarkH2DCompleteResponse resp;
    bool ok = apply_h2d_complete(req.rank_id, req.rank_epoch, req.lease_id,
                                 req.tensor_id, req.version, req.slot_id,
                                 req.keep_pinned, req.complete_seq);
    resp.ok = ok;
    resp.message = ok ? "ok" : "rejected";
    if (!ok)
        Trace::instance().event("ERROR", req.rank_id, 0, req.tensor_id, req.version,
                                req.lease_id, req.slot_id, 0);
    return resp;
}

ReleaseLeaseResponse OffloadDaemon::handle_release_lease(
    const ReleaseLeaseRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    ReleaseLeaseResponse resp;
    if (!epoch_valid(req.rank_id, req.rank_epoch)) {
        Metrics::instance().inc(Metric::kEpochMismatchRejected);
        resp.ok = false; resp.message = "epoch invalid"; return resp;
    }
    Lease* L = find_lease(req.lease_id);
    if (!L) {
        // Idempotent: already released (e.g. freed by drain).
        resp.ok = true; resp.message = "already released"; return resp;
    }
    if (L->owner_rank != req.rank_id) {
        Metrics::instance().inc(Metric::kLeaseOwnerMismatchRejected);
        resp.ok = false; resp.message = "not lease owner"; return resp;
    }
    uint32_t base = L->base_slot, cnt = L->slot_count;
    uint64_t tid = L->tensor_id, ver = L->version;
    uint32_t st = slot_state(base);

    // If this slot holds the ONLY latest copy (PINNED, no cold), releasing it
    // loses data: mark location NONE so future queries do not point at a freed
    // slot. (RACE_CONDITIONS §9 — the rank has explicitly asked to release.)
    if (st == OFLD_SLOT_PINNED_VALID || st == OFLD_SLOT_RESERVED_D2H ||
        st == OFLD_SLOT_H2D_IN_FLIGHT || st == OFLD_SLOT_D2H_IN_FLIGHT ||
        st == OFLD_SLOT_READBACK_IN_FLIGHT) {
        auto lit = locations_.find(tid);
        if (lit != locations_.end() && lit->second.slot_id == base &&
            lit->second.kind == OFLD_LOC_PINNED && lit->second.version == ver) {
            lit->second.kind = OFLD_LOC_NONE;
        }
        free_slots(base, cnt);
        leases_.erase(req.lease_id);
        Trace::instance().event("LEASE_RELEASED", req.rank_id, 0, tid, ver,
                                req.lease_id, base, 0);
        resp.ok = true; resp.message = "ok";
        return resp;
    }
    // DRAIN_IN_FLIGHT / COLD_VALID: the drain worker owns the slot lifecycle.
    leases_.erase(req.lease_id);
    Trace::instance().event("LEASE_RELEASED", req.rank_id, 0, tid, ver,
                            req.lease_id, base, 0);
    resp.ok = true; resp.message = "ok (slot managed by drain)";
    return resp;
}

LocationQueryResponse OffloadDaemon::handle_query_location(
    const LocationQueryRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    LocationQueryResponse resp;
    auto it = locations_.find(req.tensor_id);
    if (it == locations_.end()) {
        resp.ok = false; resp.message = "unknown tensor"; return resp;
    }
    const Location& loc = it->second;
    if (req.version != 0 && req.version != loc.version) {
        resp.ok = false; resp.message = "version not latest"; return resp;
    }
    resp.ok = true; resp.message = "ok";
    resp.location_kind = static_cast<uint32_t>(loc.kind);
    resp.slot_id = loc.slot_id;
    resp.nbytes = loc.nbytes;
    resp.version = loc.version;
    return resp;
}

BatchCompleteResponse OffloadDaemon::handle_batch_complete(
    const BatchCompleteRequest& req) {
    std::lock_guard<std::mutex> lk(mu_);
    BatchCompleteResponse resp;
    uint32_t accepted = 0, rejected = 0;
    for (const auto& c : req.completions) {
        bool ok = false;
        if (c.event_type == static_cast<uint32_t>(EventType::kD2HComplete)) {
            bool latest = false, enq = false;
            ok = apply_d2h_complete(req.rank_id, req.rank_epoch, c.lease_id,
                                    c.tensor_id, c.version, c.slot_id, c.seq,
                                    &latest, &enq);
        } else if (c.event_type == static_cast<uint32_t>(EventType::kH2DComplete)) {
            ok = apply_h2d_complete(req.rank_id, req.rank_epoch, c.lease_id,
                                    c.tensor_id, c.version, c.slot_id,
                                    c.keep_pinned, c.seq);
        } else {
            Metrics::instance().inc(Metric::kInvalidTransitionRejected);
            ok = false;
        }
        if (ok) ++accepted; else ++rejected;
    }
    resp.ok = true;
    resp.message = "ok";
    resp.accepted = accepted;
    resp.rejected = rejected;
    return resp;
}

GetStatsResponse OffloadDaemon::handle_get_stats(const GetStatsRequest& req) {
    (void)req;
    GetStatsResponse resp;
    resp.ok = true;
    resp.message = "ok";
    Metrics::instance().snapshot(&resp.keys, &resp.values);
    return resp;
}

ShutdownResponse OffloadDaemon::handle_shutdown(const ShutdownRequest& req) {
    (void)req;
    ShutdownResponse resp;
    resp.ok = true;
    resp.message = "shutting down";
    OFLD_INFO(TAG, "shutdown requested via RPC");
    request_stop();
    return resp;
}

// ============================================================================
// Synchronous drain-one (used to relieve pressure under the alloc lock).
// Assumes lk owns mu_. Unlocks for IO, relocks to commit. Returns true if a
// run was drained and freed (space likely available).
// ============================================================================
bool OffloadDaemon::force_drain_one(std::unique_lock<std::mutex>& lk) {
    // Find the oldest PINNED_VALID lease run.
    uint64_t best_ts = ~0ull;
    uint64_t best_lease = 0;
    for (auto& kv : leases_) {
        Lease& L = kv.second;
        if (L.is_prefetch) continue;
        if (slot_state(L.base_slot) != OFLD_SLOT_PINNED_VALID) continue;
        uint64_t ts = slots_[L.base_slot].last_touch_ns;
        if (ts < best_ts) { best_ts = ts; best_lease = L.lease_id; }
    }
    if (best_lease == 0) return false;

    Lease L = leases_[best_lease];
    for (uint32_t k = 0; k < L.slot_count; ++k)
        set_slot_state(L.base_slot + k, OFLD_SLOT_DRAIN_IN_FLIGHT);
    Metrics::instance().add(Metric::kDrainingBytes, L.nbytes);
    Trace::instance().event("DRAIN_STARTED", (uint32_t)L.owner_rank, 0, L.tensor_id,
                            L.version, L.lease_id, L.base_slot, L.nbytes);

    lk.unlock();
    bool ok = false;
    ColdEntry cold = do_drain_to_cold(L.base_slot, L.slot_count, L.nbytes, &ok);
    lk.lock();

    Metrics::instance().sub(Metric::kDrainingBytes, L.nbytes);
    if (!ok) {
        // IO failed: restore slots to PINNED_VALID (data still intact).
        for (uint32_t k = 0; k < L.slot_count; ++k)
            set_slot_state(L.base_slot + k, OFLD_SLOT_PINNED_VALID);
        Metrics::instance().inc(Metric::kIoFailure);
        return false;
    }

    uint64_t cref = next_cold_ref_++;
    cold_store_[cref] = cold;
    for (uint32_t k = 0; k < L.slot_count; ++k) slots_[L.base_slot + k].cold_ref = cref;
    set_slot_state(L.base_slot, OFLD_SLOT_COLD_VALID);
    update_location_if_latest(L.tensor_id, L.version, cold.kind, L.base_slot, cref,
                              L.nbytes);
    Trace::instance().event("COLD_VALID", (uint32_t)L.owner_rank, 0, L.tensor_id,
                            L.version, L.lease_id, L.base_slot, L.nbytes);
    free_slots(L.base_slot, L.slot_count);
    leases_.erase(best_lease);
    Metrics::instance().inc(Metric::kDrainsCompleted);
    return true;
}

// ============================================================================
// Background workers
// ============================================================================
void OffloadDaemon::drain_worker_loop() {
    while (true) {
        DrainJob job;
        {
            std::unique_lock<std::mutex> g(drain_mu_);
            drain_cv_.wait(g, [this] {
                return !drain_queue_.empty() || !workers_running_.load();
            });
            if (!workers_running_.load() && drain_queue_.empty()) return;
            job = drain_queue_.front();
            drain_queue_.pop_front();
        }

        // Snapshot metadata under mu_; verify state is DRAIN_IN_FLIGHT.
        uint64_t nbytes = 0, tensor_id = 0, version = 0, lease_id = 0;
        uint32_t owner_rank = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (slot_state(job.base_slot) != OFLD_SLOT_DRAIN_IN_FLIGHT) {
                // Slot no longer drainable (recovered/freed); skip.
                Metrics::instance().sub(Metric::kDrainingBytes,
                                        slots_[job.base_slot].nbytes);
                continue;
            }
            const ofld_slot_entry_t& e = slots_[job.base_slot];
            nbytes = e.nbytes;
            tensor_id = e.tensor_id;
            version = e.version;
            lease_id = e.lease_id;
            owner_rank = static_cast<uint32_t>(e.owner_rank);
        }

        Trace::instance().event("DRAIN_STARTED", owner_rank, 0, tensor_id, version,
                                lease_id, job.base_slot, nbytes);

        bool ok = false;
        ColdEntry cold = do_drain_to_cold(job.base_slot, job.slot_count, nbytes, &ok);

        std::lock_guard<std::mutex> lk(mu_);
        Metrics::instance().sub(Metric::kDrainingBytes, nbytes);
        if (slot_state(job.base_slot) != OFLD_SLOT_DRAIN_IN_FLIGHT) {
            // Raced with recovery; drop cold buffer if any.
            if (cold.pageable) std::free(cold.pageable);
            continue;
        }
        if (!ok) {
            for (uint32_t k = 0; k < job.slot_count; ++k)
                set_slot_state(job.base_slot + k, OFLD_SLOT_PINNED_VALID);
            Metrics::instance().inc(Metric::kIoFailure);
            continue;
        }
        uint64_t cref = next_cold_ref_++;
        cold_store_[cref] = cold;
        for (uint32_t k = 0; k < job.slot_count; ++k)
            slots_[job.base_slot + k].cold_ref = cref;
        set_slot_state(job.base_slot, OFLD_SLOT_COLD_VALID);
        update_location_if_latest(tensor_id, version, cold.kind, job.base_slot, cref,
                                  nbytes);
        Trace::instance().event("COLD_VALID", owner_rank, 0, tensor_id, version,
                                lease_id, job.base_slot, nbytes);
        free_slots(job.base_slot, job.slot_count);
        leases_.erase(lease_id);
        Metrics::instance().inc(Metric::kDrainsCompleted);
    }
}

void OffloadDaemon::readback_worker_loop() {
    while (true) {
        ReadbackJob job;
        {
            std::unique_lock<std::mutex> g(readback_mu_);
            readback_cv_.wait(g, [this] {
                return !readback_queue_.empty() || !workers_running_.load();
            });
            if (!workers_running_.load() && readback_queue_.empty()) return;
            job = readback_queue_.front();
            readback_queue_.pop_front();
        }

        ColdEntry cold;
        bool have_cold = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (slot_state(job.base_slot) != OFLD_SLOT_READBACK_IN_FLIGHT) continue;
            auto it = cold_store_.find(job.cold_ref);
            if (it != cold_store_.end()) { cold = it->second; have_cold = true; }
        }
        if (!have_cold) {
            std::lock_guard<std::mutex> lk(mu_);
            for (uint32_t k = 0; k < job.slot_count; ++k)
                set_slot_state(job.base_slot + k, OFLD_SLOT_ERROR);
            Metrics::instance().inc(Metric::kIoFailure);
            continue;
        }

        bool ok = do_readback_from_cold(job.base_slot, job.slot_count, cold,
                                        job.nbytes);

        std::lock_guard<std::mutex> lk(mu_);
        if (slot_state(job.base_slot) != OFLD_SLOT_READBACK_IN_FLIGHT) continue;
        if (!ok) {
            for (uint32_t k = 0; k < job.slot_count; ++k)
                set_slot_state(job.base_slot + k, OFLD_SLOT_ERROR);
            Metrics::instance().inc(Metric::kIoFailure);
            continue;
        }
        uint64_t tensor_id = slots_[job.base_slot].tensor_id;
        uint64_t version = slots_[job.base_slot].version;
        for (uint32_t k = 0; k < job.slot_count; ++k)
            set_slot_state(job.base_slot + k, OFLD_SLOT_PINNED_VALID);
        // Now readable in pinned; keep the cold copy too (cold_ref retained).
        update_location_if_latest(tensor_id, version, OFLD_LOC_PINNED,
                                  job.base_slot, job.cold_ref, job.nbytes);
        Metrics::instance().inc(Metric::kReadbacksCompleted);
        Trace::instance().event("PINNED_READY", 0, 0, tensor_id, version, 0,
                                job.base_slot, job.nbytes);
    }
}

void OffloadDaemon::heartbeat_monitor_loop() {
    uint64_t timeout_ns = cfg_.heartbeat_timeout_ms * 1000000ull;
    uint64_t interval_ms = std::max<uint64_t>(cfg_.heartbeat_timeout_ms / 2, 50);
    while (workers_running_.load()) {
        // Sleep in small chunks so we exit promptly on stop.
        for (uint64_t slept = 0; slept < interval_ms && workers_running_.load();
             slept += 50) {
            struct timespec ts{0, 50 * 1000000L};
            nanosleep(&ts, nullptr);
        }
        if (!workers_running_.load()) break;

        uint64_t now = now_real_ns();
        std::lock_guard<std::mutex> lk(mu_);
        header_->heartbeat_ns.store(now, std::memory_order_release);
        for (auto& kv : sessions_) {
            Session& s = kv.second;
            if (!s.alive) continue;
            if (now - s.last_heartbeat_ns <= timeout_ns) continue;

            OFLD_WARN(TAG, "rank %u epoch %llu heartbeat timeout; recovering",
                      s.rank_id, (unsigned long long)s.rank_epoch);
            s.alive = false;
            Metrics::instance().inc(Metric::kRankSessionsInvalidated);

            std::vector<uint64_t> to_erase;
            for (auto& lkv : leases_) {
                Lease& L = lkv.second;
                if (L.owner_rank != s.rank_id || L.rank_epoch != s.rank_epoch) continue;
                uint32_t st = slot_state(L.base_slot);
                if (st == OFLD_SLOT_PINNED_VALID || st == OFLD_SLOT_COLD_VALID ||
                    st == OFLD_SLOT_DRAIN_IN_FLIGHT) {
                    // Latest valid copy is safe: keep the slot, drop lease binding.
                    to_erase.push_back(L.lease_id);
                } else {
                    // In-flight, never completed: reclaim.
                    set_slot_state(L.base_slot, OFLD_SLOT_ERROR);
                    free_slots(L.base_slot, L.slot_count);
                    to_erase.push_back(L.lease_id);
                    Trace::instance().event("ERROR", s.rank_id, 0, L.tensor_id,
                                            L.version, L.lease_id, L.base_slot,
                                            L.nbytes);
                }
            }
            for (uint64_t id : to_erase) leases_.erase(id);
        }
    }
}

// ============================================================================
// Completion-ring poller (shared-memory hot path)
// ============================================================================
// Applies one completion popped from a rank's ring. mu_ must be held.
void OffloadDaemon::apply_ring_completion(uint32_t rank_id, uint64_t rank_epoch,
                                          const ofld_completion_entry_t& e) {
    if (e.event_type == static_cast<uint32_t>(EventType::kD2HComplete)) {
        bool latest = false, enq = false;
        apply_d2h_complete(rank_id, rank_epoch, e.lease_id, e.tensor_id,
                           e.version, e.slot_id, /*complete_seq=*/0, &latest, &enq);
    } else if (e.event_type == static_cast<uint32_t>(EventType::kH2DComplete)) {
        // The ring entry has no keep_pinned field, so ring H2D completions
        // default keep_pinned=false (release pinned if a cold copy exists —
        // the common restore case).
        apply_h2d_complete(rank_id, rank_epoch, e.lease_id, e.tensor_id,
                           e.version, e.slot_id, /*keep_pinned=*/false,
                           /*complete_seq=*/0);
    } else {
        Metrics::instance().inc(Metric::kInvalidTransitionRejected);
    }
}

void OffloadDaemon::ring_poller_loop() {
    if (ring_max_ranks_ == 0) return;
    const long poll_us = static_cast<long>(cfg_.ring_poll_us ? cfg_.ring_poll_us : 50);
    while (workers_running_.load()) {
        bool any = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (uint32_t i = 0; i < ring_max_ranks_; ++i) {
                if (ring_owner_.size() <= i || ring_owner_[i] == 0xFFFFFFFFu) continue;
                CompletionRingHeader* rh =
                    ring_for_index(control_base_, header_, i);
                if (!rh || rh->rank_epoch == 0) continue;
                uint32_t owner = rh->owner_rank;
                uint64_t epoch = rh->rank_epoch;
                ofld_completion_entry_t e;
                // Drain up to a bounded batch per ring per sweep to keep the
                // lock hold time short under many rings.
                int budget = 256;
                while (budget-- > 0 && ring_pop(rh, &e)) {
                    apply_ring_completion(owner, epoch, e);
                    any = true;
                }
            }
        }
        if (!any) {
            struct timespec ts { poll_us / 1000000L, (poll_us % 1000000L) * 1000L };
            nanosleep(&ts, nullptr);
        }
    }
    // Final drain on shutdown so no completion is lost.
    std::lock_guard<std::mutex> lk(mu_);
    for (uint32_t i = 0; i < ring_max_ranks_; ++i) {
        if (ring_owner_.size() <= i || ring_owner_[i] == 0xFFFFFFFFu) continue;
        CompletionRingHeader* rh = ring_for_index(control_base_, header_, i);
        if (!rh || rh->rank_epoch == 0) continue;
        ofld_completion_entry_t e;
        while (ring_pop(rh, &e)) apply_ring_completion(rh->owner_rank, rh->rank_epoch, e);
    }
}

// ============================================================================
// Cold storage IO
// ============================================================================
OffloadDaemon::ColdEntry OffloadDaemon::do_drain_to_cold(uint32_t base_slot,
                                                         uint32_t slot_count,
                                                         uint64_t nbytes,
                                                         bool* ok) {
    (void)slot_count;
    ColdEntry entry;
    entry.nbytes = nbytes;
    *ok = false;
    void* src = slot_addr(base_slot);  // contiguous run
    uint64_t t0 = now_mono_ns();

    bool want_nvme = false, want_pageable = false;
    switch (cfg_.drain_target) {
        case DrainTarget::kPageableOnly: want_pageable = true; break;
        case DrainTarget::kNvmeOnly: want_nvme = true; break;
        case DrainTarget::kPageableThenNvme: want_pageable = true; break;
    }
    if (want_nvme && !cfg_.nvme_enabled) { want_nvme = false; want_pageable = true; }

    if (want_pageable) {
        void* buf = std::malloc(nbytes);
        if (!buf) { OFLD_ERR(TAG, "drain: malloc(%llu) failed",
                             (unsigned long long)nbytes); return entry; }
        std::memcpy(buf, src, nbytes);
        entry.kind = OFLD_LOC_PAGEABLE;
        entry.pageable = buf;
        Metrics::instance().add(Metric::kPinnedToPageableBytes, nbytes);
        Metrics::instance().add(Metric::kPageableResidentBytes, nbytes);
        *ok = true;
    } else if (want_nvme) {
        if (!nvme_write(0, src, nbytes, &entry)) return entry;
        entry.kind = OFLD_LOC_NVME;
        Metrics::instance().add(Metric::kNvmeWriteBytes, nbytes);
        Metrics::instance().add(Metric::kNvmeResidentBytes, nbytes);
        *ok = true;
    }

    if (*ok) {
        uint64_t dt = now_mono_ns() - t0;
        if (dt > 0) {
            double mbps = (double)nbytes / (1024.0 * 1024.0) / ((double)dt / 1e9);
            Metrics::instance().record_bandwidth_mbps(mbps);
        }
    }
    return entry;
}

bool OffloadDaemon::do_readback_from_cold(uint32_t base_slot, uint32_t slot_count,
                                          const ColdEntry& cold, uint64_t nbytes) {
    (void)slot_count;
    void* dst = slot_addr(base_slot);
    uint64_t t0 = now_mono_ns();
    bool ok = false;
    if (cold.kind == OFLD_LOC_PAGEABLE) {
        if (!cold.pageable) return false;
        std::memcpy(dst, cold.pageable, nbytes);
        Metrics::instance().add(Metric::kPageableToPinnedBytes, nbytes);
        ok = true;
    } else if (cold.kind == OFLD_LOC_NVME) {
        if (!nvme_read(cold, dst, nbytes)) return false;
        Metrics::instance().add(Metric::kNvmeReadBytes, nbytes);
        ok = true;
    }
    if (ok) {
        uint64_t dt = now_mono_ns() - t0;
        if (dt > 0) {
            double mbps = (double)nbytes / (1024.0 * 1024.0) / ((double)dt / 1e9);
            Metrics::instance().record_bandwidth_mbps(mbps);
        }
    }
    return ok;
}

// ============================================================================
// NVMe (O_DIRECT pwrite/pread, optional striping across drives)
// ============================================================================
namespace {
// mkdir -p for a single path.
void mkdir_p(const std::string& p, const char* tag) {
    std::string cur;
    for (size_t i = 0; i <= p.size(); ++i) {
        if (i == p.size() || p[i] == '/') {
            if (!cur.empty() && cur != "/") {
                if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                    OFLD_WARN(tag, "mkdir(%s) failed: %s", cur.c_str(),
                              std::strerror(errno));
                }
            }
            if (i < p.size()) cur.push_back('/');
        } else {
            cur.push_back(p[i]);
        }
    }
}
}  // namespace

void OffloadDaemon::ensure_nvme_dir() {
    // Base path plus any per-drive stripe directories. Pointing stripe_dirs at
    // one directory per physical NVMe lets stripes land on different devices.
    mkdir_p(cfg_.nvme_path, TAG);
    for (const auto& d : cfg_.nvme_stripe_dirs) mkdir_p(d, TAG);
}

// Directory that stripe i should live in: round-robin over stripe_dirs if set,
// otherwise the single nvme_path.
std::string OffloadDaemon::stripe_dir(uint32_t stripe) const {
    if (!cfg_.nvme_stripe_dirs.empty())
        return cfg_.nvme_stripe_dirs[stripe % cfg_.nvme_stripe_dirs.size()];
    return cfg_.nvme_path;
}

bool OffloadDaemon::use_uring_engine() const {
    return cfg_.nvme_io_engine == "io_uring";
}

// Layout of a tensor across stripe files. A tensor is split into `stripes`
// contiguous segments; segment i goes to stripe file i (which may live on a
// different drive via stripe_dir). Each segment is recorded as one NvmeChunk
// whose (path, offset, length) lets the reader reconstruct it. The actual IO
// within a segment is done in <=nvme_block_bytes blocks so the daemon never
// allocates a bounce buffer larger than one block (bounded memory even for an
// 80 GB tensor), and, for io_uring, keeps up to queue_depth blocks in flight.
bool OffloadDaemon::nvme_write(uint64_t cold_ref, const void* src, uint64_t nbytes,
                               ColdEntry* out) {
    (void)cold_ref;
    if (use_uring_engine()) return nvme_write_uring(src, nbytes, out);
    return nvme_write_pwrite(src, nbytes, out);
}

bool OffloadDaemon::nvme_read(const ColdEntry& e, void* dst, uint64_t nbytes) {
    if (use_uring_engine()) return nvme_read_uring(e, dst, nbytes);
    return nvme_read_pwrite(e, dst, nbytes);
}

// Compute the per-stripe segment sizes for a tensor of nbytes. Returns a vector
// of length nvme_stripe_count_ (some entries may be 0 for tiny tensors).
std::vector<uint64_t> OffloadDaemon::stripe_segments(uint64_t nbytes) const {
    uint32_t stripes = nvme_stripe_count_;
    std::vector<uint64_t> segs(stripes, 0);
    uint64_t per = nbytes / stripes;
    uint64_t consumed = 0;
    for (uint32_t i = 0; i < stripes; ++i) {
        segs[i] = (i == stripes - 1) ? (nbytes - consumed) : per;
        consumed += segs[i];
    }
    return segs;
}

bool OffloadDaemon::nvme_write_pwrite(const void* src, uint64_t nbytes,
                                      ColdEntry* out) {
    const char* csrc = static_cast<const char*>(src);
    int open_flags = O_WRONLY | O_CREAT;
    if (cfg_.nvme_direct_io) open_flags |= O_DIRECT;
    const uint64_t block = cfg_.nvme_block_bytes ? cfg_.nvme_block_bytes
                                                 : (16ull << 20);

    // One aligned bounce block reused for every write (bounded memory).
    void* abuf = nullptr;
    if (posix_memalign(&abuf, 4096, roundup(block, 4096)) != 0) {
        OFLD_ERR(TAG, "nvme posix_memalign(%llu) failed", (unsigned long long)block);
        return false;
    }

    std::vector<uint64_t> segs = stripe_segments(nbytes);
    uint64_t consumed = 0;
    bool ok = true;
    for (uint32_t i = 0; i < segs.size() && ok; ++i) {
        uint64_t seg = segs[i];
        if (seg == 0 && i != 0) continue;

        char path[1024];
        std::snprintf(path, sizeof(path), "%s/off_stripe_%u.dat",
                      stripe_dir(i).c_str(), i);
        int fd = ::open(path, open_flags, 0644);
        if (fd < 0) {
            OFLD_ERR(TAG, "nvme open(%s) failed: %s", path, std::strerror(errno));
            ok = false; break;
        }
        // Reserve a padded region in this stripe file for the whole segment.
        uint64_t seg_padded = roundup(seg, 4096);
        uint64_t base_off = nvme_stripe_offsets_[i].fetch_add(seg_padded);

        uint64_t written_seg = 0;
        while (written_seg < seg && ok) {
            uint64_t this_block = std::min<uint64_t>(block, seg - written_seg);
            uint64_t bpad = roundup(this_block, 4096);
            if (bpad > this_block) std::memset(abuf, 0, bpad);
            std::memcpy(abuf, csrc + consumed + written_seg, this_block);
            uint64_t woff = base_off + written_seg;  // 4096-aligned (block mult)
            uint64_t w = 0;
            while (w < bpad) {
                ssize_t n = ::pwrite(fd, static_cast<char*>(abuf) + w, bpad - w,
                                     woff + w);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    OFLD_ERR(TAG, "nvme pwrite failed: %s", std::strerror(errno));
                    ok = false; break;
                }
                w += static_cast<uint64_t>(n);
            }
            written_seg += this_block;
        }
        ::close(fd);
        if (ok) {
            ColdEntry::NvmeChunk chunk;
            chunk.path = path; chunk.offset = base_off; chunk.length = seg;
            out->nvme_chunks.push_back(std::move(chunk));
            consumed += seg;
        }
    }
    std::free(abuf);
    return ok;
}

bool OffloadDaemon::nvme_read_pwrite(const ColdEntry& e, void* dst, uint64_t nbytes) {
    char* cdst = static_cast<char*>(dst);
    uint64_t consumed = 0;
    int open_flags = O_RDONLY;
    if (cfg_.nvme_direct_io) open_flags |= O_DIRECT;
    const uint64_t block = cfg_.nvme_block_bytes ? cfg_.nvme_block_bytes
                                                 : (16ull << 20);

    void* abuf = nullptr;
    if (posix_memalign(&abuf, 4096, roundup(block, 4096)) != 0) return false;

    bool ok = true;
    for (const auto& chunk : e.nvme_chunks) {
        int fd = ::open(chunk.path.c_str(), open_flags);
        if (fd < 0) {
            OFLD_ERR(TAG, "nvme open(%s) for read failed: %s", chunk.path.c_str(),
                     std::strerror(errno));
            ok = false; break;
        }
        uint64_t read_seg = 0;
        while (read_seg < chunk.length && ok) {
            uint64_t this_block = std::min<uint64_t>(block, chunk.length - read_seg);
            uint64_t bpad = roundup(this_block, 4096);
            uint64_t roff = chunk.offset + read_seg;
            uint64_t r = 0;
            while (r < bpad) {
                ssize_t n = ::pread(fd, static_cast<char*>(abuf) + r, bpad - r,
                                    roff + r);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    OFLD_ERR(TAG, "nvme pread failed: %s", std::strerror(errno));
                    ok = false; break;
                }
                if (n == 0) break;  // EOF (last block padding)
                r += static_cast<uint64_t>(n);
            }
            std::memcpy(cdst + consumed + read_seg, abuf, this_block);
            read_seg += this_block;
        }
        ::close(fd);
        if (ok) consumed += chunk.length;
    }
    std::free(abuf);
    if (ok && consumed != nbytes) {
        OFLD_WARN(TAG, "nvme_read read %llu of %llu bytes",
                  (unsigned long long)consumed, (unsigned long long)nbytes);
    }
    return true;
}

// ----------------------------------------------------------------------------
// io_uring engine (real async O_DIRECT). A tensor is split into stripe segments
// (one file per stripe, possibly per drive) and each segment into <=block_bytes
// blocks. Up to queue_depth blocks are kept in flight at once, with a fixed
// pool of queue_depth aligned bounce buffers — so memory is bounded regardless
// of tensor size, and throughput scales with queue depth + block size (both
// matter a lot for O_DIRECT, per the measured 1MB->16MB improvement).
// liburing 1.0.7 surface only.
// ----------------------------------------------------------------------------
bool OffloadDaemon::nvme_write_uring(const void* src, uint64_t nbytes,
                                     ColdEntry* out) {
    const char* csrc = static_cast<const char*>(src);
    int open_flags = O_WRONLY | O_CREAT;
    if (cfg_.nvme_direct_io) open_flags |= O_DIRECT;
    const uint64_t block = cfg_.nvme_block_bytes ? cfg_.nvme_block_bytes
                                                 : (16ull << 20);
    const uint32_t qd = std::max<uint32_t>(1, cfg_.nvme_queue_depth);

    // Open one fd per stripe and reserve a padded region for its segment.
    std::vector<uint64_t> segs = stripe_segments(nbytes);
    std::vector<int> fds(segs.size(), -1);
    std::vector<uint64_t> base_off(segs.size(), 0);
    bool ok = true;
    {
        uint64_t consumed = 0;
        for (uint32_t i = 0; i < segs.size() && ok; ++i) {
            if (segs[i] == 0 && i != 0) { continue; }
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/off_stripe_%u.dat",
                          stripe_dir(i).c_str(), i);
            fds[i] = ::open(path, open_flags, 0644);
            if (fds[i] < 0) {
                OFLD_ERR(TAG, "uring open(%s) failed: %s", path, std::strerror(errno));
                ok = false; break;
            }
            base_off[i] = nvme_stripe_offsets_[i].fetch_add(roundup(segs[i], 4096));
            ColdEntry::NvmeChunk chunk;
            chunk.path = path; chunk.offset = base_off[i]; chunk.length = segs[i];
            out->nvme_chunks.push_back(std::move(chunk));
            consumed += segs[i];
        }
    }

    // Pool of qd aligned bounce buffers.
    std::vector<void*> pool(qd, nullptr);
    for (uint32_t k = 0; k < qd && ok; ++k) {
        if (posix_memalign(&pool[k], 4096, roundup(block, 4096)) != 0) ok = false;
    }

    struct io_uring ring;
    bool ring_ok = false;
    if (ok && io_uring_queue_init(qd, &ring, 0) == 0) {
        ring_ok = true;
    } else if (ok) {
        OFLD_ERR(TAG, "io_uring_queue_init failed: %s", std::strerror(errno));
        ok = false;
    }

    // Build the flat list of block IOs across all stripes.
    struct BlockIO { int fd; uint64_t file_off; uint64_t src_off; uint64_t len; };
    std::vector<BlockIO> ios;
    if (ok) {
        uint64_t consumed = 0;
        for (uint32_t i = 0; i < segs.size(); ++i) {
            if (fds[i] < 0) continue;
            uint64_t off = 0;
            while (off < segs[i]) {
                uint64_t len = std::min<uint64_t>(block, segs[i] - off);
                ios.push_back({fds[i], base_off[i] + off, consumed + off, len});
                off += len;
            }
            consumed += segs[i];
        }
    }

    // Submit with a sliding window of qd in-flight SQEs.
    if (ok && ring_ok) {
        struct iovec iov[64];  // qd is small; guard below
        std::vector<uint32_t> buf_of_slot(qd, 0);
        size_t next = 0, done = 0, inflight = 0;
        uint32_t free_buf = 0;
        // Map cqe user_data -> buffer index via a simple queue of free buffers.
        std::vector<uint32_t> free_bufs;
        for (uint32_t k = 0; k < qd; ++k) free_bufs.push_back(k);
        std::vector<uint64_t> pad_of(ios.size(), 0);

        while (done < ios.size() && ok) {
            // Fill the pipe.
            while (inflight < qd && next < ios.size() && !free_bufs.empty()) {
                uint32_t bi = free_bufs.back(); free_bufs.pop_back();
                BlockIO& b = ios[next];
                uint64_t bpad = roundup(b.len, 4096);
                if (bpad > b.len) std::memset(pool[bi], 0, bpad);
                std::memcpy(pool[bi], csrc + b.src_off, b.len);
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) { free_bufs.push_back(bi); break; }
                iov[bi % 64].iov_base = pool[bi];
                iov[bi % 64].iov_len = bpad;
                io_uring_prep_writev(sqe, b.fd, &iov[bi % 64], 1, b.file_off);
                // Pack (block index, buffer index) into user_data.
                io_uring_sqe_set_data(sqe,
                    reinterpret_cast<void*>((static_cast<uintptr_t>(next) << 16) | bi));
                pad_of[next] = bpad;
                ++next; ++inflight;
                (void)buf_of_slot; (void)free_buf;
            }
            if (io_uring_submit(&ring) < 0) { ok = false; break; }
            struct io_uring_cqe* cqe = nullptr;
            if (io_uring_wait_cqe(&ring, &cqe) < 0) { ok = false; break; }
            uintptr_t ud = reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe));
            uint32_t bi = static_cast<uint32_t>(ud & 0xFFFF);
            uint32_t blk = static_cast<uint32_t>(ud >> 16);
            int res = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            if (res < 0 || (uint64_t)res != pad_of[blk]) {
                OFLD_ERR(TAG, "uring write cqe res=%d (want %llu)", res,
                         (unsigned long long)pad_of[blk]);
                ok = false;
            }
            free_bufs.push_back(bi);
            --inflight; ++done;
        }
    }

    if (ring_ok) io_uring_queue_exit(&ring);
    for (void* p : pool) if (p) std::free(p);
    for (int fd : fds) if (fd >= 0) ::close(fd);
    return ok;
}

bool OffloadDaemon::nvme_read_uring(const ColdEntry& e, void* dst, uint64_t nbytes) {
    int open_flags = O_RDONLY;
    if (cfg_.nvme_direct_io) open_flags |= O_DIRECT;
    const uint64_t block = cfg_.nvme_block_bytes ? cfg_.nvme_block_bytes
                                                 : (16ull << 20);
    const uint32_t qd = std::max<uint32_t>(1, cfg_.nvme_queue_depth);
    char* cdst = static_cast<char*>(dst);

    // Open each chunk's file once.
    std::vector<int> fds(e.nvme_chunks.size(), -1);
    bool ok = true;
    for (size_t i = 0; i < e.nvme_chunks.size() && ok; ++i) {
        fds[i] = ::open(e.nvme_chunks[i].path.c_str(), open_flags);
        if (fds[i] < 0) {
            OFLD_ERR(TAG, "uring open(%s) read failed: %s",
                     e.nvme_chunks[i].path.c_str(), std::strerror(errno));
            ok = false;
        }
    }

    // Flatten to block reads.
    struct BlockIO { int fd; uint64_t file_off; uint64_t dst_off; uint64_t len; };
    std::vector<BlockIO> ios;
    {
        uint64_t consumed = 0;
        for (size_t i = 0; i < e.nvme_chunks.size(); ++i) {
            const auto& c = e.nvme_chunks[i];
            uint64_t off = 0;
            while (off < c.length) {
                uint64_t len = std::min<uint64_t>(block, c.length - off);
                ios.push_back({fds[i], c.offset + off, consumed + off, len});
                off += len;
            }
            consumed += c.length;
        }
    }

    std::vector<void*> pool(qd, nullptr);
    for (uint32_t k = 0; k < qd && ok; ++k)
        if (posix_memalign(&pool[k], 4096, roundup(block, 4096)) != 0) ok = false;

    struct io_uring ring;
    bool ring_ok = false;
    if (ok && io_uring_queue_init(qd, &ring, 0) == 0) ring_ok = true;
    else if (ok) { OFLD_ERR(TAG, "io_uring_queue_init(read) failed"); ok = false; }

    if (ok && ring_ok) {
        struct iovec iov[64];
        std::vector<uint32_t> free_bufs;
        for (uint32_t k = 0; k < qd; ++k) free_bufs.push_back(k);
        std::vector<uint64_t> len_of(ios.size(), 0), dstoff_of(ios.size(), 0);
        std::vector<uint32_t> buf_of(ios.size(), 0);
        size_t next = 0, done = 0, inflight = 0;

        while (done < ios.size() && ok) {
            while (inflight < qd && next < ios.size() && !free_bufs.empty()) {
                uint32_t bi = free_bufs.back(); free_bufs.pop_back();
                BlockIO& b = ios[next];
                uint64_t bpad = roundup(b.len, 4096);
                struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) { free_bufs.push_back(bi); break; }
                iov[bi % 64].iov_base = pool[bi];
                iov[bi % 64].iov_len = bpad;
                io_uring_prep_readv(sqe, b.fd, &iov[bi % 64], 1, b.file_off);
                io_uring_sqe_set_data(sqe,
                    reinterpret_cast<void*>((static_cast<uintptr_t>(next) << 16) | bi));
                len_of[next] = b.len; dstoff_of[next] = b.dst_off; buf_of[next] = bi;
                ++next; ++inflight;
            }
            if (io_uring_submit(&ring) < 0) { ok = false; break; }
            struct io_uring_cqe* cqe = nullptr;
            if (io_uring_wait_cqe(&ring, &cqe) < 0) { ok = false; break; }
            uintptr_t ud = reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe));
            uint32_t bi = static_cast<uint32_t>(ud & 0xFFFF);
            uint32_t blk = static_cast<uint32_t>(ud >> 16);
            int res = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            // O_DIRECT read may return fewer than padded bytes at EOF; require
            // at least the logical length.
            if (res < 0 || (uint64_t)res < len_of[blk]) {
                OFLD_ERR(TAG, "uring read cqe res=%d (want >=%llu)", res,
                         (unsigned long long)len_of[blk]);
                ok = false;
            } else {
                std::memcpy(cdst + dstoff_of[blk], pool[bi], len_of[blk]);
            }
            free_bufs.push_back(bi);
            --inflight; ++done;
        }
    }

    if (ring_ok) io_uring_queue_exit(&ring);
    for (void* p : pool) if (p) std::free(p);
    for (int fd : fds) if (fd >= 0) ::close(fd);
    if (ok) {
        uint64_t total = 0;
        for (const auto& c : e.nvme_chunks) total += c.length;
        if (total != nbytes)
            OFLD_WARN(TAG, "uring nvme_read covered %llu of %llu bytes",
                      (unsigned long long)total, (unsigned long long)nbytes);
    }
    return ok;
}

}  // namespace offload
