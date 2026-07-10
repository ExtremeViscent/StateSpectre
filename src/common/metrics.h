// Metrics counters + structured JSON trace, per METRICS.md.
//
// Metrics is a fixed set of named atomic counters plus a few histograms for
// bandwidth/latency percentiles. Trace writes newline-delimited JSON events to
// a file when enabled. Both are thread-safe and lock-light on the hot path.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace offload {

// Named daemon counters. Kept as a fixed enum so GetStats can enumerate them.
enum class Metric : int {
    // transfers (bytes)
    kD2HBytes = 0,
    kH2DBytes,
    kPinnedToPageableBytes,
    kPageableToPinnedBytes,
    kNvmeWriteBytes,
    kNvmeReadBytes,
    // occupancy
    kUsedPinnedBytes,
    kInflightD2HBytes,
    kInflightH2DBytes,
    kDrainingBytes,
    kPageableResidentBytes,
    kNvmeResidentBytes,
    // counters
    kOffloadRequests,
    kLeasesGranted,
    kBlockedReservations,
    kRemoteNumaAllocs,
    kDrainsEnqueued,
    kDrainsCompleted,
    kPrefetchRequests,
    kReadbacksStarted,
    kReadbacksCompleted,
    // correctness rejections
    kInvalidTransitionRejected,
    kStaleVersionRejected,
    kEpochMismatchRejected,
    kSlotOverlapPrevented,
    kLeaseOwnerMismatchRejected,
    kChecksumMismatch,
    kIoFailure,
    kRankSessionsInvalidated,
    kMetricCount,
};

const char* metric_name(Metric m);

class Metrics {
 public:
    static Metrics& instance();

    void add(Metric m, uint64_t v) {
        c_[static_cast<int>(m)].fetch_add(v, std::memory_order_relaxed);
    }
    void inc(Metric m) { add(m, 1); }
    void sub(Metric m, uint64_t v) {
        c_[static_cast<int>(m)].fetch_sub(v, std::memory_order_relaxed);
    }
    uint64_t get(Metric m) const {
        return c_[static_cast<int>(m)].load(std::memory_order_relaxed);
    }

    // Record a completed transfer to update bandwidth histograms (MB/s buckets).
    void record_bandwidth_mbps(double mbps);
    // Percentiles over recorded bandwidth samples (p in [0,1]).
    double bandwidth_percentile(double p) const;

    // Dump all counters as parallel key/value vectors (for GetStats).
    void snapshot(std::vector<std::string>* keys, std::vector<uint64_t>* values) const;

 private:
    Metrics();
    std::atomic<uint64_t> c_[static_cast<int>(Metric::kMetricCount)];
    mutable std::mutex bw_mu_;
    std::vector<double> bw_samples_;  // ring, capped
};

// Structured trace. Enabled by daemon (--trace path) or env OFLD_TRACE.
class Trace {
 public:
    static Trace& instance();

    void enable(const std::string& path);
    void disable();
    bool enabled() const { return fp_ != nullptr; }

    // Emit one event. Fields match METRICS.md recommended schema.
    void event(const char* name, uint32_t rank, uint32_t gpu,
               uint64_t tensor_id, uint64_t version, uint64_t lease_id,
               uint32_t slot_id, uint64_t nbytes);

 private:
    Trace() = default;
    ~Trace();
    std::mutex mu_;
    FILE* fp_ = nullptr;
};

}  // namespace offload
