// Rank-side pseudo-code sketch. Not intended to compile as-is.

#include "offload_abi.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

struct ArenaMapping {
    ofld_arena_id_t arena_id;
    void* base;
    uint64_t size;
};

struct InflightD2H {
    ofld_lease_id_t lease_id;
    ofld_tensor_id_t tensor_id;
    ofld_version_t version;
    ofld_slot_id_t slot_id;
    cudaEvent_t event;
    void* tensor_to_invalidate; // placeholder for PyTorch Tensor handle/ref
};

class RankAgent {
public:
    void register_rank();
    void mmap_arenas();
    void register_cuda_chunks();

    void* slot_ptr(const ofld_slot_entry_t& slot) {
        return static_cast<char*>(arena_base_[slot.arena_id]) + slot.arena_offset;
    }

    OffloadHandle evict_tensor(TorchTensor tensor, const char* name, cudaStream_t stream) {
        validate_tensor_for_destructive_evict(tensor);

        auto meta = capture_tensor_metadata(tensor);
        auto lease = rpc_request_offload(meta.tensor_id, meta.version, meta.nbytes,
                                         gpu_id_, numa_node_);

        auto& slot = slot_table_[lease.slot_id];
        void* dst = slot_ptr(slot);
        void* src = tensor.data_ptr();

        rpc_mark_d2h_submitted(lease);

        cudaMemcpyAsync(dst, src, meta.nbytes, cudaMemcpyDeviceToHost, stream);

        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);

        inflight_.push_back({lease.lease_id, meta.tensor_id, meta.version,
                             lease.slot_id, ev, tensor.raw_handle()});

        return OffloadHandle{meta, lease};
    }

    void progress_once() {
        for (auto it = inflight_.begin(); it != inflight_.end();) {
            cudaError_t st = cudaEventQuery(it->event);
            if (st == cudaSuccess) {
                rpc_mark_d2h_complete(*it);
                replace_tensor_storage_with_empty(it->tensor_to_invalidate);
                cudaEventDestroy(it->event);
                it = inflight_.erase(it);
            } else if (st == cudaErrorNotReady) {
                ++it;
            } else {
                rpc_mark_error(*it, st);
                ++it;
            }
        }
    }

    TorchTensor restore(const OffloadHandle& h, cudaStream_t stream) {
        auto lease = rpc_request_prefetch(h.tensor_id, h.version, h.nbytes,
                                          gpu_id_, numa_node_);
        wait_until_prefetch_ready(lease);

        TorchTensor out = allocate_cuda_tensor(h.shape, h.dtype, h.device);
        auto& slot = slot_table_[lease.slot_id];
        void* src = slot_ptr(slot);
        void* dst = out.data_ptr();

        rpc_mark_h2d_submitted(lease);
        cudaMemcpyAsync(dst, src, h.nbytes, cudaMemcpyHostToDevice, stream);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);

        track_h2d_completion(lease, ev);
        return out;
    }

private:
    int gpu_id_;
    int numa_node_;
    std::unordered_map<ofld_arena_id_t, void*> arena_base_;
    ofld_slot_entry_t* slot_table_;
    std::vector<InflightD2H> inflight_;
};
