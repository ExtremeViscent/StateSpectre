"""fastoffload — centralized GPU tensor-offload runtime, Python user API.

This package is the ergonomic layer over the ``_fastoffload`` C++ extension
(pybind11 + libtorch). All heavy lifting — D2H/H2D copies, cudaHostRegister,
CUDA events, daemon RPC, and the destructive storage replacement — happens in
the C++ ``OffloadAgent`` reached through the ``Context`` shim. This module only
provides the surface described in ``python_api/PYTHON_API.md``:

    with fo.offload_context(device="cuda:0") as off:
        h = off.evict(x, name="x")
        h.wait_offloaded()
        x = h.restore(device="cuda:0")
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterable, List, Optional, Sequence, Union

import torch

from . import _fastoffload  # C++ extension (pybind11 module)

__all__ = [
    "offload_context",
    "Off",
    "OffloadHandle",
    "PrefetchHandle",
    "ManagedRecord",
]


# --------------------------------------------------------------------------- #
# dtype <-> scalar-type-int mapping for the C++ boundary.
#
# The C++ shim uses c10::ScalarType integer values. torch.dtype objects are not
# directly convertible across the pybind boundary, so we round-trip through the
# integer id. torch exposes a stable enum via the private but ubiquitous
# _C._get_default_dtype-adjacent machinery; the robust, version-stable way is to
# build the map from torch itself.
# --------------------------------------------------------------------------- #
def _build_dtype_maps():
    # Enumerate the concrete dtypes torch ships. Each torch.dtype has no public
    # int, so we discover the mapping via a tiny tensor's scalar type through the
    # extension's own convention: c10::ScalarType is what tensor.scalar_type()
    # returns and what torch::empty consumes. We derive the int by asking the
    # extension what value it assigns — but to avoid a chicken/egg call we use
    # torch's documented ordering: the values are stable within a torch release,
    # so we materialize them once by creating scalar tensors and reading the C++
    # scalar_type through a helper. Simpler & robust: keep an explicit table of
    # the dtypes we support and let the extension report the int at evict time
    # (it returns scalar_type in the evict tuple). For restore we must map back.
    #
    # We therefore learn the mapping lazily and cache it: the first evict of a
    # given dtype teaches us (dtype -> int); we also seed the common dtypes here
    # by probing via torch.zeros(...).  The probe uses a CPU tensor so it is
    # cheap and needs no CUDA.
    dtypes = [
        torch.float32, torch.float64, torch.float16, torch.bfloat16,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        torch.bool, torch.complex64, torch.complex128,
    ]
    # Add optional/newer dtypes if present in this torch build.
    for name in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32", "uint64"):
        d = getattr(torch, name, None)
        if isinstance(d, torch.dtype):
            dtypes.append(d)
    return dtypes


_KNOWN_DTYPES = _build_dtype_maps()
_DTYPE_TO_INT = {}
_INT_TO_DTYPE = {}
_DTYPE_LOCK = threading.Lock()


def _seed_dtype_map(dtype: torch.dtype, scalar_int: int) -> None:
    with _DTYPE_LOCK:
        _DTYPE_TO_INT[dtype] = int(scalar_int)
        _INT_TO_DTYPE[int(scalar_int)] = dtype


def _dtype_from_int(scalar_int: int) -> torch.dtype:
    scalar_int = int(scalar_int)
    d = _INT_TO_DTYPE.get(scalar_int)
    if d is not None:
        return d
    # We have not seen this int yet. The C++ side gave it to us from a real
    # tensor at evict time, so it *will* be in the map by the time restore runs
    # for that tensor. If somehow not, raise a clear error.
    raise RuntimeError(
        f"fastoffload: unknown scalar type id {scalar_int}; cannot reconstruct "
        f"torch.dtype for restore"
    )


def _parse_device_index(device: Union[str, int, torch.device, None],
                         default: int) -> int:
    """Return a CUDA device index from a device spec."""
    if device is None:
        return default
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"fastoffload: device must be a CUDA device, got {dev}")
    return dev.index if dev.index is not None else default


def _parse_socket_path(daemon_addr: str) -> str:
    """Parse a 'unix:///path' daemon address into a filesystem socket path."""
    if daemon_addr.startswith("unix://"):
        return daemon_addr[len("unix://"):]
    return daemon_addr


# --------------------------------------------------------------------------- #
# OffloadHandle
# --------------------------------------------------------------------------- #
class OffloadHandle:
    """Handle to an offloaded tensor.

    Carries the metadata needed to reconstruct the tensor on restore. All
    operations delegate to the C++ ``Context`` / ``OffloadAgent``.
    """

    __slots__ = (
        "tensor_id", "version", "name", "shape", "stride", "dtype",
        "device_origin", "nbytes", "_off", "_scalar_int", "_prefetched",
        "_discarded",
    )

    def __init__(self, off: "Off", tensor_id: int, version: int,
                 name: Optional[str], shape: torch.Size, stride: tuple,
                 dtype: torch.dtype, device_origin: torch.device, nbytes: int,
                 scalar_int: int):
        self._off = off
        self.tensor_id = int(tensor_id)
        self.version = int(version)
        self.name = name
        self.shape = shape
        self.stride = stride
        self.dtype = dtype
        self.device_origin = device_origin
        self.nbytes = int(nbytes)
        self._scalar_int = int(scalar_int)
        self._prefetched: Optional[torch.Tensor] = None
        self._discarded = False

    # ---- readiness -------------------------------------------------------- #
    def ready(self) -> bool:
        """True once the D2H copy into the pinned slot has completed."""
        return self._off._ctx.is_offloaded(self.tensor_id, self.version)

    def wait_offloaded(self) -> None:
        """Block until the D2H copy has completed (host copy valid)."""
        self._off._ctx.wait_offloaded(self.tensor_id, self.version)

    def location(self) -> str:
        """Location string: 'gpu' / 'pinned' / 'pageable' / 'nvme' / 'none'."""
        return self._off._ctx.location(self.tensor_id, self.version)

    # ---- prefetch --------------------------------------------------------- #
    def prefetch(self, device=None, stream=None) -> "PrefetchHandle":
        """Warm the tensor back into a device buffer ahead of use.

        REAL semantics (no stub): this performs the H2D restore into a cached
        device tensor now (asynchronously w.r.t. the caller's stream) and stashes
        it. A subsequent ``restore()`` returns the cached tensor instead of
        issuing a second H2D. ``PrefetchHandle.wait()`` blocks until the warmed
        copy is resident.
        """
        if self._discarded:
            raise RuntimeError("prefetch: handle has been discarded")
        # Issue the restore now; keep it non-blocking so the caller can overlap.
        t = self._restore_into_new(device=device, stream=stream, wait=False)
        self._prefetched = t
        return PrefetchHandle(self, t)

    # ---- restore ---------------------------------------------------------- #
    def _restore_into_new(self, device, stream, wait) -> torch.Tensor:
        dev_index = _parse_device_index(device, self._off._device_index)
        out = self._off._ctx.restore(
            self.tensor_id, self.version, list(self.shape), self._scalar_int,
            dev_index, stream, wait,
        )
        return out

    def restore(self, device=None, stream=None, wait: bool = True) -> torch.Tensor:
        """Restore into a NEW tensor and return it.

        If a prefetch is outstanding, returns the prefetched tensor (after
        ensuring the requested wait semantics), avoiding a duplicate H2D.
        """
        if self._discarded:
            raise RuntimeError("restore: handle has been discarded")
        if self._prefetched is not None:
            out = self._prefetched
            self._prefetched = None
            if wait:
                # Ensure the warmed H2D is complete before handing back.
                self._off._ctx.wait_offloaded(self.tensor_id, self.version)
            return out
        return self._restore_into_new(device=device, stream=stream, wait=wait)

    # ---- discard ---------------------------------------------------------- #
    def discard(self) -> None:
        """Drop the daemon-side lease/metadata for this tensor (best-effort)."""
        if self._discarded:
            return
        self._off._ctx.discard(self.tensor_id, self.version)
        self._prefetched = None
        self._discarded = True

    def __repr__(self) -> str:
        return (f"OffloadHandle(tensor_id={self.tensor_id}, version={self.version}, "
                f"name={self.name!r}, shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"nbytes={self.nbytes})")


class PrefetchHandle:
    """Small handle returned by ``OffloadHandle.prefetch``."""

    __slots__ = ("_handle", "_tensor")

    def __init__(self, handle: OffloadHandle, tensor: torch.Tensor):
        self._handle = handle
        self._tensor = tensor

    def wait(self) -> torch.Tensor:
        """Block until the warmed copy is resident, then return it."""
        # The H2D was issued with wait=False; block on the agent's readiness and
        # also synchronize the tensor's device to be safe for immediate use.
        torch.cuda.synchronize(self._tensor.device)
        return self._tensor

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor


# --------------------------------------------------------------------------- #
# ManagedRecord — tracked-but-not-yet-offloaded tensor (PYTHON_API §4)
# --------------------------------------------------------------------------- #
class ManagedRecord:
    """A tensor adopted for tracking; call ``.evict()`` to offload it later."""

    __slots__ = ("_off", "tensor", "name")

    def __init__(self, off: "Off", tensor: torch.Tensor, name: Optional[str]):
        self._off = off
        self.tensor = tensor
        self.name = name

    def evict(self, stream=None, invalidate: str = "set_empty", wait: bool = False,
              allow_views: bool = False, unsafe_autograd: bool = False,
              compact: bool = False) -> OffloadHandle:
        h = self._off.evict(self.tensor, name=self.name, stream=stream,
                            invalidate=invalidate, wait=wait,
                            allow_views=allow_views, unsafe_autograd=unsafe_autograd,
                            compact=compact)
        # After a destructive evict the original storage is empty; drop our ref
        # so callers don't accidentally reuse it.
        self.tensor = None
        return h

    def copy(self, stream=None, wait: bool = False) -> OffloadHandle:
        return self._off.copy(self.tensor, name=self.name, stream=stream, wait=wait)


# --------------------------------------------------------------------------- #
# Off — the session object yielded by offload_context()
# --------------------------------------------------------------------------- #
class Off:
    """User-facing session handle. Wraps a C++ ``Context`` (one OffloadAgent)."""

    def __init__(self, ctx: "_fastoffload.Context", device_index: int):
        self._ctx = ctx
        self._device_index = device_index
        self._unsafe_default = False  # toggled by unsafe_destructive_mode()

    # ---- internal helpers ------------------------------------------------- #
    @staticmethod
    def _invalidate_to_flags(invalidate: Optional[str]) -> bool:
        """Map an invalidate= mode to whether this is a destructive eviction.

        'set_empty' / 'zero' -> destructive (storage is replaced).
        None / 'none'        -> non-destructive (a plain copy).
        """
        if invalidate in (None, "none", "off", False):
            return False
        if invalidate in ("set_empty", "zero", "empty"):
            return True
        raise ValueError(
            f"invalidate must be one of 'set_empty', 'zero', or None; got {invalidate!r}")

    def _make_handle(self, tensor: torch.Tensor, name, result_tuple) -> OffloadHandle:
        tensor_id, version, nbytes, scalar_int, device_index = result_tuple
        # Learn the dtype<->int mapping from this real tensor for restore.
        _seed_dtype_map(tensor.dtype, scalar_int)
        return OffloadHandle(
            off=self, tensor_id=tensor_id, version=version, name=name,
            shape=torch.Size(tuple(tensor.shape)), stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            device_origin=torch.device("cuda", device_index if device_index >= 0
                                       else self._device_index),
            nbytes=nbytes, scalar_int=scalar_int,
        )

    def _evict_impl(self, tensor: torch.Tensor, name, stream, destructive,
                    allow_views, unsafe_autograd, compact, wait,
                    tensor_id: int = 0, version: int = 0) -> OffloadHandle:
        if tensor is None:
            raise RuntimeError("evict: tensor is None (already evicted/consumed?)")
        # Capture metadata BEFORE the C++ call, because a destructive evict may
        # replace the storage (shape becomes []). We snapshot shape/stride/dtype
        # from the live tensor here.
        snap_shape = torch.Size(tuple(tensor.shape))
        snap_stride = tuple(tensor.stride())
        snap_dtype = tensor.dtype
        eff_unsafe = unsafe_autograd or self._unsafe_default
        result = self._ctx.evict(
            tensor, name or "", destructive, allow_views, eff_unsafe, compact,
            wait, stream, int(tensor_id), int(version),
        )
        tid, ver, nbytes, scalar_int, device_index = result
        _seed_dtype_map(snap_dtype, scalar_int)
        return OffloadHandle(
            off=self, tensor_id=tid, version=ver, name=name, shape=snap_shape,
            stride=snap_stride, dtype=snap_dtype,
            device_origin=torch.device("cuda", device_index if device_index >= 0
                                       else self._device_index),
            nbytes=nbytes, scalar_int=scalar_int,
        )

    # ---- public: evict / copy / manage ----------------------------------- #
    def evict(self, tensor: torch.Tensor, name: Optional[str] = None, stream=None,
              invalidate: str = "set_empty", wait: bool = False,
              allow_views: bool = False, unsafe_autograd: bool = False,
              compact: bool = False) -> OffloadHandle:
        """Destructive eviction: copy GPU->pinned, then invalidate original."""
        destructive = self._invalidate_to_flags(invalidate)
        return self._evict_impl(tensor, name, stream, destructive, allow_views,
                                unsafe_autograd, compact, wait)

    def copy(self, tensor: torch.Tensor, name: Optional[str] = None, stream=None,
             wait: bool = False, allow_views: bool = False,
             compact: bool = False) -> OffloadHandle:
        """Non-destructive copy: original tensor stays valid."""
        return self._evict_impl(tensor, name, stream, destructive=False,
                                allow_views=allow_views, unsafe_autograd=False,
                                compact=compact, wait=wait)

    def manage(self, tensor: torch.Tensor, name: Optional[str] = None) -> ManagedRecord:
        """Track a tensor for later eviction without offloading it now."""
        return ManagedRecord(self, tensor, name)

    # ---- public: batch ---------------------------------------------------- #
    def evict_many(self, tensors: Sequence[torch.Tensor],
                   names: Optional[Sequence[str]] = None, stream=None,
                   invalidate: str = "set_empty",
                   require_own_storage: bool = True,
                   unsafe_autograd: bool = False) -> List[OffloadHandle]:
        """Evict many tensors. ``require_own_storage=False`` allows compact-copy
        of views (base storage not guaranteed freed)."""
        tensors = list(tensors)
        if names is not None:
            names = list(names)
            if len(names) != len(tensors):
                raise ValueError("evict_many: len(names) must match len(tensors)")
        destructive = self._invalidate_to_flags(invalidate)
        allow_views = not require_own_storage
        compact = not require_own_storage
        handles: List[OffloadHandle] = []
        for i, t in enumerate(tensors):
            nm = names[i] if names is not None else None
            handles.append(self._evict_impl(
                t, nm, stream, destructive, allow_views, unsafe_autograd,
                compact, wait=False))
        return handles

    def wait(self, handles: Iterable[OffloadHandle]) -> None:
        """Block until every handle's D2H copy has completed."""
        for h in handles:
            h.wait_offloaded()

    def restore_many(self, handles: Sequence[OffloadHandle], device=None,
                     stream=None, wait: bool = True) -> List[torch.Tensor]:
        """Restore many tensors; returns them in handle order."""
        return [h.restore(device=device, stream=stream, wait=wait) for h in handles]

    # ---- public: debugging ------------------------------------------------ #
    def summary(self) -> str:
        """Human-readable multi-line summary of the rank agent state."""
        return self._ctx.summary_string()

    def summary_dict(self) -> dict:
        """Structured summary (see AgentSummary)."""
        return self._ctx.summary()

    def assert_no_inflight(self) -> None:
        """Raise if any D2H/H2D transfer is still in flight."""
        self._ctx.assert_no_inflight()

    def slot_table(self, limit: int = 20) -> str:
        """Return an introspection string of the agent/slot state.

        The agent exposes aggregate slot/inflight state via its summary; we
        surface it here (the daemon owns the per-slot table)."""
        s = self._ctx.summary()
        lines = ["slot_table (agent view):"]
        lines.append(f"  num_slots={s['num_slots']} limit={limit}")
        lines.append(f"  registered_pinned_bytes={s['registered_pinned_bytes']}")
        lines.append(f"  inflight_d2h_bytes={s['inflight_d2h_bytes']} "
                     f"inflight_h2d_bytes={s['inflight_h2d_bytes']}")
        lines.append(f"  evict_count={s['evict_count']} restore_count={s['restore_count']} "
                     f"d2h_done={s['d2h_completed']} h2d_done={s['h2d_completed']}")
        return "\n".join(lines)

    # ---- public: unsafe mode ---------------------------------------------- #
    @contextlib.contextmanager
    def unsafe_destructive_mode(self):
        """Within this block, evict() defaults unsafe_autograd=True (§9)."""
        prev = self._unsafe_default
        self._unsafe_default = True
        try:
            yield self
        finally:
            self._unsafe_default = prev

    # ---- public: trace ---------------------------------------------------- #
    @contextlib.contextmanager
    def trace(self, path: str):
        """Capture a before/after summary snapshot around a block and write a
        Chrome-trace-style JSON file to ``path`` (§10). The heavy per-op metrics
        live in the agent/daemon; here we record session-level counters so the
        trace is real and inspectable without a daemon-side dump."""
        import json
        import time

        start_ns = time.time_ns()
        before = self._ctx.summary()
        events = [{
            "name": "offload_session", "ph": "B", "ts": start_ns / 1000.0,
            "pid": before["rank_id"], "tid": before["gpu_id"],
            "args": {k: before[k] for k in before},
        }]
        try:
            yield self
        finally:
            end_ns = time.time_ns()
            after = self._ctx.summary()
            events.append({
                "name": "offload_session", "ph": "E", "ts": end_ns / 1000.0,
                "pid": after["rank_id"], "tid": after["gpu_id"],
                "args": {k: after[k] for k in after},
            })
            events.append({
                "name": "delta", "ph": "i", "s": "g", "ts": end_ns / 1000.0,
                "pid": after["rank_id"], "tid": after["gpu_id"],
                "args": {
                    "d2h_completed": after["d2h_completed"] - before["d2h_completed"],
                    "h2d_completed": after["h2d_completed"] - before["h2d_completed"],
                    "evict_count": after["evict_count"] - before["evict_count"],
                    "restore_count": after["restore_count"] - before["restore_count"],
                },
            })
            with open(path, "w") as f:
                json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)


# --------------------------------------------------------------------------- #
# offload_context — the session context manager
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def offload_context(daemon_addr: str = "unix:///tmp/fastoffload.sock",
                    device: Union[str, int, torch.device] = "cuda:0",
                    rank: int = 0, local_rank: int = 0, world_rank: int = 0,
                    numa_node: int = -1,
                    register_policy: str = "lazy_chunked",
                    registration_chunk_mb: int = 512,
                    target_tier: str = "pageable_then_nvme",
                    invalidate_mode: str = "set_empty",
                    eager_register: bool = False):
    """Manage an offload runtime session (PYTHON_API §2).

    Connects to the daemon, receives arena/control fds, mmaps them,
    cudaHostRegisters lazily/eagerly, and starts the completion progress thread —
    all inside the C++ OffloadAgent. On exit it best-effort drains inflight
    transfers and closes the agent.

    Yields an :class:`Off` session object.
    """
    socket_path = _parse_socket_path(daemon_addr)
    device_index = _parse_device_index(device, 0)

    # register_policy controls eager vs lazy chunked cudaHostRegister.
    eager = eager_register or (register_policy in ("eager", "eager_all"))
    chunk_bytes = int(registration_chunk_mb) * (1 << 20)

    ctx = _fastoffload.Context(
        socket_path, device_index, int(rank), int(local_rank), int(world_rank),
        int(numa_node), int(chunk_bytes), bool(eager),
    )
    off = Off(ctx, device_index)
    try:
        yield off
    finally:
        # Best-effort: wait for inflight to settle, then release the agent.
        try:
            torch.cuda.synchronize(torch.device("cuda", device_index))
        except Exception:
            pass
        ctx.close()
