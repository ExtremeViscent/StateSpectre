#include "metrics.h"

#include <algorithm>
#include <cmath>

#include "util.h"

namespace offload {

const char* metric_name(Metric m) {
    switch (m) {
        case Metric::kD2HBytes: return "d2h_bytes";
        case Metric::kH2DBytes: return "h2d_bytes";
        case Metric::kPinnedToPageableBytes: return "pinned_to_pageable_bytes";
        case Metric::kPageableToPinnedBytes: return "pageable_to_pinned_bytes";
        case Metric::kNvmeWriteBytes: return "nvme_write_bytes";
        case Metric::kNvmeReadBytes: return "nvme_read_bytes";
        case Metric::kUsedPinnedBytes: return "used_pinned_bytes";
        case Metric::kInflightD2HBytes: return "inflight_d2h_bytes";
        case Metric::kInflightH2DBytes: return "inflight_h2d_bytes";
        case Metric::kDrainingBytes: return "draining_bytes";
        case Metric::kPageableResidentBytes: return "pageable_resident_bytes";
        case Metric::kNvmeResidentBytes: return "nvme_resident_bytes";
        case Metric::kOffloadRequests: return "offload_requests";
        case Metric::kLeasesGranted: return "leases_granted";
        case Metric::kBlockedReservations: return "blocked_reservation_count";
        case Metric::kRemoteNumaAllocs: return "remote_numa_alloc_count";
        case Metric::kDrainsEnqueued: return "drains_enqueued";
        case Metric::kDrainsCompleted: return "drains_completed";
        case Metric::kPrefetchRequests: return "prefetch_requests";
        case Metric::kReadbacksStarted: return "readbacks_started";
        case Metric::kReadbacksCompleted: return "readbacks_completed";
        case Metric::kInvalidTransitionRejected: return "invalid_transition_rejected";
        case Metric::kStaleVersionRejected: return "stale_version_completion_rejected";
        case Metric::kEpochMismatchRejected: return "rank_epoch_mismatch_rejected";
        case Metric::kSlotOverlapPrevented: return "slot_overlap_prevented";
        case Metric::kLeaseOwnerMismatchRejected: return "lease_owner_mismatch_rejected";
        case Metric::kChecksumMismatch: return "checksum_mismatch";
        case Metric::kIoFailure: return "io_failure";
        case Metric::kRankSessionsInvalidated: return "rank_sessions_invalidated";
        case Metric::kMetricCount: return "metric_count";
    }
    return "unknown";
}

Metrics& Metrics::instance() {
    static Metrics m;
    return m;
}

Metrics::Metrics() {
    for (auto& c : c_) c.store(0, std::memory_order_relaxed);
    bw_samples_.reserve(4096);
}

void Metrics::record_bandwidth_mbps(double mbps) {
    if (!(mbps > 0.0) || std::isinf(mbps)) return;
    std::lock_guard<std::mutex> g(bw_mu_);
    if (bw_samples_.size() >= 8192) {
        // keep a rolling window: drop oldest half
        bw_samples_.erase(bw_samples_.begin(), bw_samples_.begin() + 4096);
    }
    bw_samples_.push_back(mbps);
}

double Metrics::bandwidth_percentile(double p) const {
    std::lock_guard<std::mutex> g(bw_mu_);
    if (bw_samples_.empty()) return 0.0;
    std::vector<double> v = bw_samples_;
    std::sort(v.begin(), v.end());
    if (p <= 0) return v.front();
    if (p >= 1) return v.back();
    size_t idx = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return v[idx];
}

void Metrics::snapshot(std::vector<std::string>* keys,
                       std::vector<uint64_t>* values) const {
    keys->clear();
    values->clear();
    for (int i = 0; i < static_cast<int>(Metric::kMetricCount); ++i) {
        keys->push_back(metric_name(static_cast<Metric>(i)));
        values->push_back(c_[i].load(std::memory_order_relaxed));
    }
    // Append bandwidth percentiles as derived metrics.
    auto push_pct = [&](const char* k, double p) {
        keys->push_back(k);
        values->push_back(static_cast<uint64_t>(bandwidth_percentile(p)));
    };
    push_pct("bandwidth_mbps_p50", 0.50);
    push_pct("bandwidth_mbps_p90", 0.90);
    push_pct("bandwidth_mbps_p99", 0.99);
}

// -------------------- Trace --------------------

Trace& Trace::instance() {
    static Trace t;
    return t;
}

Trace::~Trace() {
    if (fp_) { std::fclose(fp_); fp_ = nullptr; }
}

void Trace::enable(const std::string& path) {
    std::lock_guard<std::mutex> g(mu_);
    if (fp_) std::fclose(fp_);
    fp_ = std::fopen(path.c_str(), "w");
    if (!fp_) {
        OFLD_WARN("trace", "could not open trace file %s", path.c_str());
    } else {
        OFLD_INFO("trace", "tracing to %s", path.c_str());
    }
}

void Trace::disable() {
    std::lock_guard<std::mutex> g(mu_);
    if (fp_) { std::fclose(fp_); fp_ = nullptr; }
}

void Trace::event(const char* name, uint32_t rank, uint32_t gpu,
                  uint64_t tensor_id, uint64_t version, uint64_t lease_id,
                  uint32_t slot_id, uint64_t nbytes) {
    if (!fp_) return;
    std::lock_guard<std::mutex> g(mu_);
    if (!fp_) return;
    std::fprintf(fp_,
        "{\"ts_ns\":%llu,\"rank\":%u,\"gpu\":%u,\"tensor_id\":%llu,"
        "\"version\":%llu,\"lease_id\":%llu,\"slot_id\":%u,\"event\":\"%s\","
        "\"nbytes\":%llu}\n",
        (unsigned long long)now_real_ns(), rank, gpu,
        (unsigned long long)tensor_id, (unsigned long long)version,
        (unsigned long long)lease_id, slot_id, name,
        (unsigned long long)nbytes);
    std::fflush(fp_);
}

}  // namespace offload
