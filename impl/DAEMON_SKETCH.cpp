// Daemon-side pseudo-code sketch. Not intended to compile as-is.

#include "offload_abi.h"
#include <mutex>
#include <unordered_map>
#include <queue>

struct LeaseRecord {
    ofld_lease_id_t lease_id;
    ofld_slot_id_t slot_id;
    ofld_tensor_id_t tensor_id;
    ofld_version_t version;
    uint64_t owner_rank;
    uint64_t owner_pid;
    uint64_t rank_epoch;
    uint64_t nbytes;
};

struct LocationRecord {
    ofld_tensor_id_t tensor_id;
    ofld_version_t version;
    ofld_location_kind_t kind;
    ofld_slot_id_t slot_id;
    uint64_t cold_ref;
    uint64_t nbytes;
};

class OffloadDaemon {
public:
    RequestOffloadResponse RequestOffload(const RequestOffloadRequest& req) {
        std::lock_guard<std::mutex> g(mu_);
        validate_rank_epoch(req.rank_id, req.rank_epoch);

        ofld_slot_id_t slot_id = allocate_slot(req.gpu_id, req.numa_node, req.nbytes);
        auto lease_id = next_lease_id_++;

        auto& slot = slot_table_[slot_id];
        slot.state = OFLD_SLOT_RESERVED_D2H;
        slot.tensor_id = req.tensor_id;
        slot.version = req.version;
        slot.lease_id = lease_id;
        slot.owner_rank = req.rank_id;
        slot.owner_pid = req.pid;
        slot.rank_epoch = req.rank_epoch;
        slot.nbytes = req.nbytes;

        leases_[lease_id] = LeaseRecord{lease_id, slot_id, req.tensor_id, req.version,
                                        req.rank_id, req.pid, req.rank_epoch, req.nbytes};

        return make_offload_response(slot, lease_id);
    }

    MarkD2HSubmittedResponse MarkD2HSubmitted(const MarkD2HSubmittedRequest& req) {
        std::lock_guard<std::mutex> g(mu_);
        auto& lease = checked_lease(req.lease_id, req.rank_id, req.rank_epoch);
        auto& slot = slot_table_[lease.slot_id];
        require_state(slot, OFLD_SLOT_RESERVED_D2H);
        slot.state = OFLD_SLOT_D2H_IN_FLIGHT;
        slot.submit_seq = req.submit_seq;
        return {true, "ok"};
    }

    MarkD2HCompleteResponse MarkD2HComplete(const MarkD2HCompleteRequest& req) {
        std::lock_guard<std::mutex> g(mu_);
        auto& lease = checked_lease(req.lease_id, req.rank_id, req.rank_epoch);
        auto& slot = slot_table_[lease.slot_id];
        require_state(slot, OFLD_SLOT_D2H_IN_FLIGHT);
        require_tensor_version(lease, req.tensor_id, req.version);

        slot.state = OFLD_SLOT_PINNED_VALID;
        slot.complete_seq = req.complete_seq;

        bool latest = update_location_if_latest(req.tensor_id, req.version,
                                                OFLD_LOC_PINNED, req.slot_id,
                                                0, lease.nbytes);

        bool enqueued = false;
        if (should_drain(slot)) {
            slot.state = OFLD_SLOT_DRAIN_IN_FLIGHT;
            drain_queue_.push(req.slot_id);
            enqueued = true;
        }

        return {true, "ok", latest, enqueued};
    }

    void drain_worker_loop() {
        while (running_) {
            ofld_slot_id_t slot_id = pop_drain_queue();
            auto& slot = slot_table_[slot_id];

            // Copy pinned slot to pageable DRAM or write to NVMe.
            // Do not hold allocator mutex while doing large IO/copy.
            ColdRef cold = drain_slot_to_cold_storage(slot);

            std::lock_guard<std::mutex> g(mu_);
            if (slot.state == OFLD_SLOT_DRAIN_IN_FLIGHT) {
                slot.cold_ref = cold.ref;
                slot.state = OFLD_SLOT_COLD_VALID;
                update_location_if_latest(slot.tensor_id, slot.version,
                                          cold.kind, slot_id, cold.ref, slot.nbytes);
                release_pinned_slot(slot_id);
            }
        }
    }

private:
    std::mutex mu_;
    ofld_slot_entry_t* slot_table_;
    std::unordered_map<ofld_lease_id_t, LeaseRecord> leases_;
    std::unordered_map<ofld_tensor_id_t, LocationRecord> locations_;
    std::queue<ofld_slot_id_t> drain_queue_;
    ofld_lease_id_t next_lease_id_ = 1;
    bool running_ = true;
};
