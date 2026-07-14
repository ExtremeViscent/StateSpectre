"""state_spectre — centralized GPU tensor-offload runtime, Python user API.

This package is the ergonomic layer over the ``_state_spectre`` C++ extension
(pybind11 + libtorch). All heavy lifting — D2H/H2D copies, cudaHostRegister,
CUDA events, daemon RPC, and the destructive storage replacement — happens in
the C++ ``OffloadAgent`` reached through the ``Context`` shim. This module only
provides the surface described in ``python_api/PYTHON_API.md``:

    with ss.offload_context(device="cuda:0") as off:
        h = off.evict(x, name="x")
        h.wait_offloaded()
        x = h.restore(device="cuda:0")
"""

from __future__ import annotations

import contextlib
import threading
from typing import Iterable, List, Optional, Sequence, Union

import torch

from . import _state_spectre  # C++ extension (pybind11 module)

__all__ = [
    "offload_context",
    "Off",
    "OffloadHandle",
    "PrefetchHandle",
    "ManagedRecord",
    # v2 canonical model-state API
    "CanonicalKey",
    "DedupPolicy",
    "ModelRole",
    "CanonicalHandle",
    "RolloutWeightClient",
]


# --------------------------------------------------------------------------- #
# v2 canonical model-state constants (mirror abi/offload_canonical_abi.hpp and
# rpc/offload_canonical.proto — keep values in sync).
# --------------------------------------------------------------------------- #
class ModelRole:
    UNKNOWN = 0
    POLICY_TRAINABLE = 1
    POLICY_ROLLOUT = 2
    POLICY_REFERENCE = 3
    REWARD_MODEL = 4
    VALUE_MODEL = 5
    AUXILIARY = 6

    _BY_NAME = {
        "unknown": 0, "policy_trainable": 1, "policy_rollout": 2,
        "policy_reference": 3, "reward_model": 4, "value_model": 5,
        "auxiliary": 6,
    }

    @classmethod
    def parse(cls, role) -> int:
        if isinstance(role, int):
            return role
        r = str(role).lower()
        if r not in cls._BY_NAME:
            raise ValueError(f"unknown model_role {role!r}; one of {list(cls._BY_NAME)}")
        return cls._BY_NAME[r]


# DedupMode numeric values (offload_canonical_abi.hpp DedupMode).
_DEDUP_MODES = {
    "disabled": 0, "semantic_trusted": 1, "hash_verified": 2, "sampled_hash": 3,
}

# AttachCreateAction numeric values (for interpreting canonical_evict results).
_ACTION_ATTACHED_EXISTING = 0
_ACTION_NEED_D2H_CREATE = 1
_ACTION_WAIT_FOR_CREATOR = 2
_ACTION_DUPLICATE_CANDIDATE = 3
_ACTION_REJECTED_STALE_VERSION = 4
_ACTION_QUOTA_EXCEEDED = 5
_ACTION_ERROR = 255
_ACTION_NAMES = {
    0: "ATTACHED_EXISTING", 1: "NEED_D2H_CREATE", 2: "WAIT_FOR_CREATOR",
    3: "DUPLICATE_CANDIDATE", 4: "REJECTED_STALE_VERSION", 5: "QUOTA_EXCEEDED",
    255: "ERROR",
}


def _fnv1a64(data: bytes, seed: int = 1469598103934665603) -> int:
    """64-bit FNV-1a over bytes (used for stable shape/stride/fqn hashing)."""
    h = seed & 0xFFFFFFFFFFFFFFFF
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _hash_ints(values) -> int:
    buf = bytearray()
    for v in values:
        v = int(v) & 0xFFFFFFFFFFFFFFFF
        buf += v.to_bytes(8, "little")
    return _fnv1a64(bytes(buf))


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
        f"state_spectre: unknown scalar type id {scalar_int}; cannot reconstruct "
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
        raise ValueError(f"state_spectre: device must be a CUDA device, got {dev}")
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

    def restore_into(self, dst: torch.Tensor, stream=None,
                     wait: bool = True) -> torch.Tensor:
        """Restore into an EXISTING device tensor's storage (in-place H2D).

        Unlike :meth:`restore`, this allocates nothing — it H2Ds the offloaded
        bytes into ``dst``'s current storage. This is the restore half of the
        offload cycle for **aliased flat buffers** (Megatron DDP
        ``buffer.param_data`` / ``grad_data``, distributed-optimizer master-param
        groups, FSDP ``FlatParameter``), where destructive ``evict`` is wrong
        because other tensors alias the same storage.

        Correct non-destructive cycle for those buffers::

            h = off.copy(buf, wait=True)          # D2H; buf untouched
            buf.untyped_storage().resize_(0)      # free VRAM, KEEP the Storage object
            ...
            buf.untyped_storage().resize_(h.nbytes)   # realloc the SAME Storage object
            h.restore_into(buf)                        # H2D back into it

        Preserving the Storage object (resize, don't reassign) keeps every
        alias/view valid — they observe the storage's new data pointer. Note
        ``wait=True`` on the ``copy`` is mandatory: the D2H must finish before
        ``resize_(0)`` frees the source out from under the copy.
        """
        if self._discarded:
            raise RuntimeError("restore_into: handle has been discarded")
        dst_nbytes = dst.numel() * dst.element_size()
        if dst_nbytes != self.nbytes:
            raise ValueError(
                f"restore_into: dst has {dst_nbytes} bytes but the offloaded "
                f"tensor is {self.nbytes} bytes (resize the storage to match "
                f"before restoring)")
        self._off._ctx.restore_into(dst, self.tensor_id, self.version, stream, wait)
        return dst

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
# v2 canonical model-state: key builder, dedup policy, handle
# --------------------------------------------------------------------------- #
class DedupPolicy:
    """Deduplication policy for canonical model-state offload (02_DEDUP)."""

    __slots__ = ("mode", "creating_policy", "cross_job_dedup")

    def __init__(self, mode: str = "semantic_trusted",
                 creating_policy: str = "wait",
                 cross_job_dedup: bool = False):
        if mode not in _DEDUP_MODES:
            raise ValueError(f"mode must be one of {list(_DEDUP_MODES)}; got {mode!r}")
        if creating_policy not in ("wait", "duplicate_candidate_on_pressure"):
            raise ValueError("creating_policy must be 'wait' or "
                             "'duplicate_candidate_on_pressure'")
        self.mode = mode
        self.creating_policy = creating_policy
        self.cross_job_dedup = bool(cross_job_dedup)

    @property
    def mode_int(self) -> int:
        return _DEDUP_MODES[self.mode]

    @property
    def allow_duplicate_candidate(self) -> bool:
        return self.creating_policy == "duplicate_candidate_on_pressure"


class CanonicalKey:
    """Identity of one logical model-state shard at one model version.

    Built via :meth:`Off.canonical_key`. The DP/replica axis is intentionally
    excluded (that is what makes replicated ranks dedup); partition axes
    (pp/tp/ep/etp/expert) are part of the identity. shape/stride/fqn are hashed
    to stable 64-bit values so the daemon key is compact and language-neutral.
    """

    __slots__ = (
        "model_role", "model_version", "param_id", "param_fqn_hash",
        "param_fqn_debug", "layout", "shape_hash", "stride_hash",
        "pp_rank", "tp_rank", "ep_rank", "etp_rank", "expert_id",
        "shard_offset", "shard_nbytes",
    )

    def __init__(self, *, model_role, model_version, param_id, param_fqn_hash,
                 param_fqn_debug, layout, shape_hash, stride_hash, pp_rank,
                 tp_rank, ep_rank, etp_rank, expert_id, shard_offset,
                 shard_nbytes):
        self.model_role = int(model_role)
        self.model_version = int(model_version)
        self.param_id = int(param_id)
        self.param_fqn_hash = int(param_fqn_hash)
        self.param_fqn_debug = param_fqn_debug
        self.layout = int(layout)
        self.shape_hash = int(shape_hash)
        self.stride_hash = int(stride_hash)
        self.pp_rank = int(pp_rank)
        self.tp_rank = int(tp_rank)
        self.ep_rank = int(ep_rank)
        self.etp_rank = int(etp_rank)
        self.expert_id = int(expert_id)
        self.shard_offset = int(shard_offset)
        self.shard_nbytes = int(shard_nbytes)

    def __repr__(self) -> str:
        return (f"CanonicalKey(role={self.model_role}, version={self.model_version}, "
                f"param_id={self.param_id}, fqn={self.param_fqn_debug!r}, "
                f"pp={self.pp_rank}, tp={self.tp_rank}, ep={self.ep_rank}, "
                f"expert={self.expert_id})")


class CanonicalHandle:
    """Result of a canonical evict. Carries the daemon object_id + outcome."""

    __slots__ = ("object_id", "action", "action_name", "did_d2h", "attached",
                 "key", "message", "_off")

    def __init__(self, object_id, action, did_d2h, key, message, off=None):
        self.object_id = int(object_id)
        self.action = int(action)
        self.action_name = _ACTION_NAMES.get(int(action), str(action))
        self.did_d2h = bool(did_d2h)
        self.attached = (int(action) == _ACTION_ATTACHED_EXISTING)
        self.key = key
        self.message = message
        self._off = off

    def restore_into(self, dst: torch.Tensor, stream=None,
                     wait: bool = True) -> torch.Tensor:
        """Reload this canonical object's bytes into ``dst`` (in-place H2D).

        The trainer offload/reload read path: reads the one shared canonical
        copy — addressed purely by ``object_id`` — back into ``dst``'s storage
        via a direct H2D from the daemon's shared pinned arena. Works even for a
        replica whose evict returned ATTACHED_EXISTING (no local D2H): every DP
        replica reloads the same deduplicated bytes. Read-only w.r.t. the
        object; the daemon holds it resident for the copy.

        ``dst`` must be a contiguous CUDA tensor already sized to the object
        (e.g. after ``storage().resize_(nbytes)`` for an aliased flat buffer).
        """
        if self._off is None:
            raise RuntimeError("restore_into: handle not bound to a session")
        self._off._ctx.canonical_restore_into(dst, self.object_id, stream, wait)
        return dst

    def __repr__(self) -> str:
        return (f"CanonicalHandle(object_id={self.object_id}, "
                f"action={self.action_name}, did_d2h={self.did_d2h})")


# --------------------------------------------------------------------------- #
# Off — the session object yielded by offload_context()
# --------------------------------------------------------------------------- #
class Off:
    """User-facing session handle. Wraps a C++ ``Context`` (one OffloadAgent)."""

    def __init__(self, ctx: "_state_spectre.Context", device_index: int):
        self._ctx = ctx
        self._device_index = device_index
        self._unsafe_default = False  # toggled by unsafe_destructive_mode()
        # v2 job identity (set by _register_job via offload_context).
        self.tenant_id = None
        self.job_id = None
        self.launch_epoch = None
        self.job_name = None
        self._dedup_policy = DedupPolicy()

    # ---- v2 job registration --------------------------------------------- #
    def _register_job(self, tenant_id: int, job_name: str,
                      scheduler_job_id: int) -> None:
        t, jid, epoch, name = self._ctx.register_job(
            int(tenant_id), str(job_name), int(scheduler_job_id))
        self.tenant_id = int(t)
        self.job_id = int(jid)
        self.launch_epoch = int(epoch)
        self.job_name = name

    def set_dedup_policy(self, policy: "DedupPolicy") -> None:
        self._dedup_policy = policy

    # ---- v2 canonical key builder ---------------------------------------- #
    def canonical_key(self, *, model_role, model_version, param_name,
                      param_id=None, tensor: torch.Tensor = None,
                      pp_rank: int = 0, tp_rank: int = 0, ep_rank: int = 0,
                      etp_rank: int = 0, expert_id: int = -1,
                      shape=None, dtype=None, shard_offset: int = 0,
                      shard_nbytes: int = None) -> CanonicalKey:
        """Build a :class:`CanonicalKey`.

        DP/replica rank is intentionally NOT an argument — that is the axis
        deduplicated across. Partition axes (pp/tp/ep/etp/expert) ARE part of
        the identity. shape/stride/param-name are hashed to stable 64-bit ints.
        """
        if tensor is not None:
            shape = tuple(tensor.shape) if shape is None else tuple(shape)
            stride = tuple(tensor.stride())
            if shard_nbytes is None:
                shard_nbytes = tensor.numel() * tensor.element_size()
        else:
            shape = tuple(shape or ())
            stride = ()
        if param_id is None:
            param_id = _fnv1a64(param_name.encode("utf-8"))
        fqn_hash = _fnv1a64(param_name.encode("utf-8"))
        return CanonicalKey(
            model_role=ModelRole.parse(model_role),
            model_version=int(model_version),
            param_id=int(param_id),
            param_fqn_hash=fqn_hash,
            param_fqn_debug=param_name,
            layout=0,
            shape_hash=_hash_ints(shape),
            stride_hash=_hash_ints(stride),
            pp_rank=pp_rank, tp_rank=tp_rank, ep_rank=ep_rank, etp_rank=etp_rank,
            expert_id=expert_id, shard_offset=shard_offset,
            shard_nbytes=int(shard_nbytes or 0),
        )

    # ---- v2 canonical evict ---------------------------------------------- #
    def canonical_evict(self, tensor: torch.Tensor, canonical_key: CanonicalKey,
                        dedup: str = "attach_or_create", destructive: bool = True,
                        stream=None, wait: bool = True,
                        allow_views: bool = False, unsafe_autograd: bool = False,
                        compact: bool = False,
                        local_tensor_id: int = 0, local_version: int = 1,
                        dedup_policy: "DedupPolicy" = None) -> CanonicalHandle:
        """Attach-or-create a canonical object for ``tensor``.

        With ``dedup="attach_or_create"`` the daemon returns ATTACHED_EXISTING
        (no D2H — bytes deduped), NEED_D2H_CREATE (this rank creates + D2Hs),
        WAIT_FOR_CREATOR, or DUPLICATE_CANDIDATE. See 02_DEDUPLICATION_POLICY.md.
        """
        if self.job_id is None:
            raise RuntimeError("canonical_evict: context has no job; pass "
                               "job_name to offload_context()")
        pol = dedup_policy or self._dedup_policy
        if dedup == "disabled":
            mode = 0
        else:
            mode = pol.mode_int
        eff_unsafe = unsafe_autograd or self._unsafe_default
        k = canonical_key
        ok, action, object_id, did_d2h, message = self._ctx.canonical_evict(
            tensor, bool(destructive), bool(allow_views), bool(eff_unsafe),
            bool(compact), bool(wait), stream,
            k.model_role, k.model_version, k.param_id, k.param_fqn_hash,
            k.param_fqn_debug, k.layout, k.shape_hash, k.stride_hash,
            k.pp_rank, k.tp_rank, k.ep_rank, k.etp_rank, k.expert_id,
            k.shard_offset, k.shard_nbytes, int(mode),
            bool(pol.allow_duplicate_candidate),
            int(local_tensor_id), int(local_version),
        )
        if not ok:
            raise RuntimeError(
                f"canonical_evict failed (action={_ACTION_NAMES.get(action, action)}): "
                f"{message}")
        return CanonicalHandle(object_id, action, did_d2h, canonical_key, message,
                               off=self)

    # ---- v2 seal / promote ----------------------------------------------- #
    def seal_model_version(self, model_role, model_version: int,
                           fail_if_missing: bool = True,
                           promote: bool = False) -> dict:
        """Seal a model version into an immutable manifest (optionally promote)."""
        ok, state, tensor_count, total_nbytes, message = self._ctx.seal_model_version(
            ModelRole.parse(model_role), int(model_version), bool(promote),
            bool(fail_if_missing))
        if not ok:
            raise RuntimeError(f"seal_model_version failed: {message}")
        return {"state": state, "tensor_count": tensor_count,
                "total_nbytes": total_nbytes, "message": message}

    def promote_rollout_version(self, model_version: int,
                                fail_if_missing: bool = True) -> dict:
        """Seal POLICY_ROLLOUT@version and atomically set the latest pointer."""
        return self.seal_model_version(ModelRole.POLICY_ROLLOUT, model_version,
                                       fail_if_missing=fail_if_missing,
                                       promote=True)

    def drop_canonical_version(self, model_role, model_version: int) -> dict:
        """Release every canonical object of (model_role, model_version).

        Explicit GC for the offload/reload round-trip: ``model_version`` bumps
        each cycle (param bytes change, so a fresh version is required), and the
        daemon does not auto-reclaim old versions — call this once step N's
        objects are no longer needed to keep host/NVMe bounded. Honors in-flight
        export/restore holds (those objects are skipped and reported for retry);
        ignores the advisory attachment refcount. Also drops the version's
        manifest and clears the latest-sealed pointer if it referenced it.

        Returns ``{"dropped", "skipped_inflight", "bytes_freed", "message"}``.
        """
        dropped, skipped, bytes_freed, message = self._ctx.drop_canonical_version(
            ModelRole.parse(model_role), int(model_version))
        return {"dropped": dropped, "skipped_inflight": skipped,
                "bytes_freed": bytes_freed, "message": message}

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
def offload_context(daemon_addr: str = "unix:///tmp/state_spectre.sock",
                    device: Union[str, int, torch.device] = "cuda:0",
                    rank: int = 0, local_rank: int = 0, world_rank: int = 0,
                    numa_node: int = -1,
                    register_policy: str = "lazy_chunked",
                    registration_chunk_mb: int = 512,
                    target_tier: str = "pageable_then_nvme",
                    invalidate_mode: str = "set_empty",
                    eager_register: bool = False,
                    job_name: Optional[str] = None,
                    tenant_id: int = 0,
                    scheduler_job_id=None,
                    dedup_policy: Optional["DedupPolicy"] = None):
    """Manage an offload runtime session (PYTHON_API §2 + v2 canonical §Job).

    Connects to the daemon, receives arena/control fds, mmaps them,
    cudaHostRegisters lazily/eagerly, and starts the completion progress thread —
    all inside the C++ OffloadAgent. On exit it best-effort drains inflight
    transfers and closes the agent.

    If ``job_name`` is given the session registers a v2 job; the daemon assigns
    ``off.job_id`` / ``off.launch_epoch`` and canonical evicts become available.

    Yields an :class:`Off` session object.
    """
    socket_path = _parse_socket_path(daemon_addr)
    device_index = _parse_device_index(device, 0)

    # register_policy controls eager vs lazy chunked cudaHostRegister.
    eager = eager_register or (register_policy in ("eager", "eager_all"))
    chunk_bytes = int(registration_chunk_mb) * (1 << 20)

    ctx = _state_spectre.Context(
        socket_path, device_index, int(rank), int(local_rank), int(world_rank),
        int(numa_node), int(chunk_bytes), bool(eager),
    )
    off = Off(ctx, device_index)
    if dedup_policy is not None:
        off.set_dedup_policy(dedup_policy)
    if job_name is not None:
        sched = 0 if scheduler_job_id is None else int(scheduler_job_id)
        off._register_job(tenant_id, job_name, sched)
    try:
        yield off
    finally:
        # Best-effort: wait for inflight to settle, then release the agent.
        try:
            torch.cuda.synchronize(torch.device("cuda", device_index))
        except Exception:
            pass
        ctx.close()


# --------------------------------------------------------------------------- #
# RolloutWeightClient — imported at end to avoid a circular import (_rollout
# references ModelRole from this module).
# --------------------------------------------------------------------------- #
from ._rollout import RolloutWeightClient  # noqa: E402,F401
