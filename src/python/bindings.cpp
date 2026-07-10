// pybind11 module `_fastoffload`: the C++ shim that bridges torch::Tensor <->
// the CUDA/torch-free offload::OffloadAgent (see src/agent/agent.h).
//
// Architecture boundary (do NOT violate): agent.h speaks in raw uint64_t device
// pointers and uintptr_t stream handles. This translation unit is the ONLY place
// that knows about torch::Tensor. It:
//   * pulls the raw device pointer from tensor.data_ptr(),
//   * resolves the CUDA stream (current stream, or a torch.cuda.Stream's
//     .cuda_stream attribute),
//   * validates destructive-eviction preconditions (PYTHON_API §7,
//     RACE_CONDITIONS §7/§8),
//   * builds the InvalidateCallback that performs REAL storage replacement
//     (DESIGN_SPEC §10: replace storage, not just shape) via Tensor::set_data,
//   * hands everything to agent->evict()/restore().
//
// The heavy lifting (D2H/H2D, cudaHostRegister, events, RPC) lives entirely in
// the agent; nothing here re-implements it.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "agent.h"

namespace py = pybind11;
using offload::AgentConfig;
using offload::AgentSummary;
using offload::InvalidateCallback;
using offload::OffloadAgent;
using offload::OffloadTicket;
using offload::RestoreResult;
using offload::StreamHandle;
using offload::TensorMeta;

namespace {

// Auto-assigned tensor_ids are namespaced per rank inside Context
// (rank in the top 24 bits, a per-process counter in the low 40) so concurrent
// processes never collide in the daemon's global (tensor_id -> location) table.
// tensor_id 0 is reserved for "assign new"; callers may pass an explicit
// (tensor_id, version) to re-evict a logical tensor at a bumped version.

// Resolve a Python stream argument to the agent's opaque StreamHandle.
//   None                       -> current CUDA stream on `device`
//   torch.cuda.Stream (or any) -> its integer `.cuda_stream` attribute
StreamHandle resolve_stream(const py::object& stream_obj, int device) {
    if (stream_obj.is_none()) {
        cudaStream_t s = c10::cuda::getCurrentCUDAStream(
                             static_cast<c10::DeviceIndex>(device))
                             .stream();
        return reinterpret_cast<StreamHandle>(s);
    }
    // torch.cuda.Stream exposes the raw cudaStream_t as an int in .cuda_stream.
    if (py::hasattr(stream_obj, "cuda_stream")) {
        auto v = stream_obj.attr("cuda_stream").cast<std::uintptr_t>();
        return static_cast<StreamHandle>(v);
    }
    // Allow passing a raw integer stream handle directly.
    try {
        auto v = stream_obj.cast<std::uintptr_t>();
        return static_cast<StreamHandle>(v);
    } catch (const py::cast_error&) {
        throw std::runtime_error(
            "stream must be None, a torch.cuda.Stream (with .cuda_stream), or an int handle");
    }
}

// Validate that a tensor is safe for DESTRUCTIVE, full-storage eviction.
// (PYTHON_API §7, DESIGN_SPEC §9, RACE_CONDITIONS §7/§8.) Throws with a precise
// message on violation. This is the full-ownership path; the compact/allow_views
// path bypasses these checks by cloning a contiguous copy instead.
void validate_destructive(const torch::Tensor& t, bool unsafe_autograd) {
    if (!t.is_cuda()) {
        throw std::runtime_error("evict: tensor must be a CUDA tensor");
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(
            "evict: tensor must be contiguous for destructive eviction "
            "(use allow_views=True, compact=True to copy a view)");
    }
    if (t.storage_offset() != 0) {
        throw std::runtime_error(
            "evict: tensor.storage_offset() must be 0 for destructive eviction "
            "(use allow_views=True, compact=True to copy a view)");
    }
    const int64_t logical_nbytes =
        static_cast<int64_t>(t.numel()) * t.element_size();
    const int64_t storage_nbytes =
        static_cast<int64_t>(t.storage().nbytes());
    if (logical_nbytes != storage_nbytes) {
        throw std::runtime_error(
            "evict: logical nbytes (" + std::to_string(logical_nbytes) +
            ") != underlying storage nbytes (" + std::to_string(storage_nbytes) +
            "); tensor does not own its full storage. Use allow_views=True, "
            "compact=True to copy the logical bytes instead.");
    }
    if (t.requires_grad() && !unsafe_autograd) {
        throw std::runtime_error(
            "evict: tensor.requires_grad is True; destructive eviction would "
            "invalidate an autograd-tracked tensor. Pass unsafe_autograd=True "
            "(inside off.unsafe_destructive_mode()) if you accept responsibility "
            "for autograd correctness (RACE_CONDITIONS §8).");
    }
}

// The C++-owned context: one OffloadAgent per offload_context() session.
class Context {
 public:
    Context(std::string socket_path, int device, int rank, int local_rank,
            int world_rank, int numa_node, std::uint64_t registration_chunk_bytes,
            bool eager_register) {
        AgentConfig cfg;
        cfg.socket_path = std::move(socket_path);
        cfg.cuda_device = device;
        cfg.rank_id = static_cast<std::uint32_t>(rank);
        cfg.local_rank = static_cast<std::uint32_t>(local_rank);
        cfg.world_rank = static_cast<std::uint32_t>(world_rank);
        cfg.numa_node = numa_node;
        if (registration_chunk_bytes != 0) {
            cfg.registration_chunk_bytes = registration_chunk_bytes;
        }
        cfg.eager_register = eager_register;
        // OffloadAgent ctor connects to the daemon, mmaps arenas, starts threads.
        agent_.reset(new OffloadAgent(cfg));
        // Namespace auto-assigned tensor_ids by rank so that concurrent
        // processes never collide in the daemon's GLOBAL (tensor_id -> location)
        // table. rank_id is unique per live session (daemon-enforced), so we
        // put it in the top 24 bits and use a per-process counter in the low 40.
        // 40 bits = ~1.1e12 evictions per process before wraparound.
        rank_id_ = static_cast<std::uint32_t>(rank);
    }

    std::uint64_t next_tensor_id() {
        std::uint64_t local = local_tensor_ctr_.fetch_add(1, std::memory_order_relaxed);
        return (static_cast<std::uint64_t>(rank_id_ & 0xFFFFFFull) << 40) |
               (local & 0xFFFFFFFFFFull);
    }

    // ---- eviction ---------------------------------------------------------
    // Returns (tensor_id, version, nbytes, scalar_type_int, device_index).
    py::tuple evict(torch::Tensor t, std::string name, bool destructive,
                    bool allow_views, bool unsafe_autograd, bool compact,
                    bool wait, py::object stream_obj, std::uint64_t tensor_id,
                    std::uint64_t version) {
        if (!t.is_cuda()) {
            throw std::runtime_error("evict: tensor must be a CUDA tensor");
        }

        const int device = agent_->cuda_device();
        const std::uint64_t scalar_type = static_cast<std::uint64_t>(t.scalar_type());
        const int device_index = t.get_device();

        // Assign identity if the caller did not pass an explicit (id, version).
        if (tensor_id == 0) {
            tensor_id = next_tensor_id();
        }
        if (version == 0) version = 1;

        // Determine the concrete device tensor we will D2H from, and whether we
        // must invalidate it after the copy.
        //
        //  * compact / allow_views: copy a fresh CONTIGUOUS CLONE of the logical
        //    region. This owns its own storage (safe to invalidate) but does NOT
        //    free the caller's base storage (documented; RACE_CONDITIONS §7).
        //  * otherwise: full-ownership destructive path — validate, then evict
        //    the original tensor's storage and (if destructive) replace it with
        //    an empty CUDA storage once the D2H event fires.
        torch::Tensor source;
        bool do_invalidate = destructive;
        if (compact || allow_views) {
            // A clone is always freshly-allocated, contiguous, offset 0.
            source = t.clone(at::MemoryFormat::Contiguous);
        } else {
            validate_destructive(t, unsafe_autograd);
            source = t;
        }

        const std::uint64_t nbytes =
            static_cast<std::uint64_t>(source.numel()) * source.element_size();
        const std::uint64_t dev_ptr =
            reinterpret_cast<std::uint64_t>(source.data_ptr());

        const StreamHandle stream = resolve_stream(stream_obj, device);

        TensorMeta meta;
        meta.tensor_id = tensor_id;
        meta.version = version;
        meta.nbytes = nbytes;
        meta.name = std::move(name);

        // Build the invalidate/keep-alive callback. Capturing `source` by value
        // keeps the device buffer alive for the entire D2H window (the agent
        // stores this std::function inside the inflight record until the copy
        // completes) — this is the free-after-event guard from RACE_CONDITIONS
        // §1, and it applies to BOTH destructive and non-destructive copies.
        //
        // When destructive, after the D2H event completes the agent invokes this
        // from its progress thread WITHOUT holding the GIL, so we acquire it via
        // gil_scoped_acquire (PyGILState_Ensure semantics — reentrant-safe, also
        // correct for the inline wait=true path). We then perform the REAL
        // storage replacement: set_data() swaps in a zero-sized CUDA tensor,
        // releasing the old storage to the caching allocator (DESIGN_SPEC §10 —
        // replace storage, not just shape).
        InvalidateCallback cb =
            [source, do_invalidate](std::uint64_t /*cookie*/) mutable {
                // Guard against interpreter teardown: never touch Python while
                // finalizing.
                if (Py_IsFinalizing()) {
                    source = torch::Tensor();
                    return;
                }
                py::gil_scoped_acquire gil;
                if (do_invalidate) {
                    source.set_data(torch::empty({0}, source.options()));
                }
                source = torch::Tensor();  // drop the keep-alive reference
            };

        OffloadTicket ticket;
        {
            // Release the GIL so (a) the agent's progress thread can grab it for
            // the callback when wait=false, and (b) other Python threads run
            // while the (possibly blocking, wait=true) D2H proceeds.
            py::gil_scoped_release unlock;
            ticket = agent_->evict(dev_ptr, nbytes, stream, destructive, meta, cb,
                                   /*cookie=*/tensor_id, wait);
        }
        if (!ticket.ok) {
            throw std::runtime_error("evict failed: " + ticket.message);
        }

        return py::make_tuple(ticket.tensor_id, ticket.version, nbytes,
                              scalar_type, device_index);
    }

    // ---- restore ----------------------------------------------------------
    // Allocates a fresh CUDA tensor of the given shape/dtype/device and asks the
    // agent to H2D the offloaded bytes into it. Returns the new tensor.
    torch::Tensor restore(std::uint64_t tensor_id, std::uint64_t version,
                          std::vector<std::int64_t> shape, std::uint64_t scalar_type,
                          int device_index, py::object stream_obj, bool wait) {
        auto dtype = static_cast<c10::ScalarType>(scalar_type);
        if (device_index < 0) device_index = agent_->cuda_device();

        auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .device(torch::Device(torch::kCUDA,
                                                 static_cast<c10::DeviceIndex>(device_index)));
        torch::Tensor out = torch::empty(shape, options);

        const std::uint64_t dev_ptr = reinterpret_cast<std::uint64_t>(out.data_ptr());
        const std::uint64_t nbytes =
            static_cast<std::uint64_t>(out.numel()) * out.element_size();
        const StreamHandle stream = resolve_stream(stream_obj, device_index);

        RestoreResult r;
        {
            py::gil_scoped_release unlock;
            r = agent_->restore(tensor_id, version, dev_ptr, nbytes, stream, wait);
        }
        if (!r.ok) {
            throw std::runtime_error("restore failed: " + r.message);
        }
        return out;
    }

    // ---- introspection / lifecycle ---------------------------------------
    bool is_offloaded(std::uint64_t tensor_id, std::uint64_t version) const {
        return agent_->is_offloaded(tensor_id, version);
    }

    void wait_offloaded(std::uint64_t tensor_id, std::uint64_t version) {
        py::gil_scoped_release unlock;
        agent_->wait_offloaded(tensor_id, version);
    }

    std::string location(std::uint64_t tensor_id, std::uint64_t version) const {
        py::gil_scoped_release unlock;
        return agent_->location(tensor_id, version);
    }

    void discard(std::uint64_t tensor_id, std::uint64_t version) {
        py::gil_scoped_release unlock;
        agent_->discard(tensor_id, version);
    }

    std::string summary_string() const {
        py::gil_scoped_release unlock;
        return agent_->summary_string();
    }

    py::dict summary() const {
        AgentSummary s = agent_->summary();
        py::dict d;
        d["rank_id"] = s.rank_id;
        d["gpu_id"] = s.gpu_id;
        d["numa_node"] = s.numa_node;
        d["registered_pinned_bytes"] = s.registered_pinned_bytes;
        d["inflight_d2h_bytes"] = s.inflight_d2h_bytes;
        d["inflight_h2d_bytes"] = s.inflight_h2d_bytes;
        d["evict_count"] = s.evict_count;
        d["restore_count"] = s.restore_count;
        d["d2h_completed"] = s.d2h_completed;
        d["h2d_completed"] = s.h2d_completed;
        d["rank_epoch"] = s.rank_epoch;
        d["num_slots"] = s.num_slots;
        return d;
    }

    void assert_no_inflight() const { agent_->assert_no_inflight(); }

    int cuda_device() const { return agent_->cuda_device(); }
    int numa_node() const { return agent_->numa_node(); }
    std::uint32_t rank_id() const { return agent_->rank_id(); }

    // Release the agent (joins threads, drains inflight, unmaps). Idempotent.
    void close() { agent_.reset(); }

 private:
    std::unique_ptr<OffloadAgent> agent_;
    std::uint32_t rank_id_ = 0;
    std::atomic<std::uint64_t> local_tensor_ctr_{1};
};

}  // namespace

PYBIND11_MODULE(_fastoffload, m) {
    m.doc() = "fastoffload C++ shim: torch::Tensor <-> offload::OffloadAgent";

    py::class_<Context>(m, "Context")
        .def(py::init<std::string, int, int, int, int, int, std::uint64_t, bool>(),
             py::arg("socket_path"), py::arg("device"), py::arg("rank"),
             py::arg("local_rank"), py::arg("world_rank"), py::arg("numa_node"),
             py::arg("registration_chunk_bytes"), py::arg("eager_register"))
        .def("evict", &Context::evict, py::arg("tensor"), py::arg("name"),
             py::arg("destructive"), py::arg("allow_views"),
             py::arg("unsafe_autograd"), py::arg("compact"), py::arg("wait"),
             py::arg("stream"), py::arg("tensor_id"), py::arg("version"))
        .def("restore", &Context::restore, py::arg("tensor_id"), py::arg("version"),
             py::arg("shape"), py::arg("scalar_type"), py::arg("device_index"),
             py::arg("stream"), py::arg("wait"))
        .def("is_offloaded", &Context::is_offloaded, py::arg("tensor_id"),
             py::arg("version"))
        .def("wait_offloaded", &Context::wait_offloaded, py::arg("tensor_id"),
             py::arg("version"))
        .def("location", &Context::location, py::arg("tensor_id"), py::arg("version"))
        .def("discard", &Context::discard, py::arg("tensor_id"), py::arg("version"))
        .def("summary_string", &Context::summary_string)
        .def("summary", &Context::summary)
        .def("assert_no_inflight", &Context::assert_no_inflight)
        .def("cuda_device", &Context::cuda_device)
        .def("numa_node", &Context::numa_node)
        .def("rank_id", &Context::rank_id)
        .def("close", &Context::close);
}
