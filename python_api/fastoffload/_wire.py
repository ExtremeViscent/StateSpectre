"""Pure-Python little-endian codec for the v2 canonical control-plane wire
protocol, mirroring src/common/wire.cpp + wire_v2.cpp.

Only the messages a *rollout engine* needs are implemented here (it talks to
the daemon's TCP control port to discover + pull sealed weights): the frame
envelope plus GetLatestSealedVersion / GetManifest / PullTensor. The trainer
side uses the C++ agent, not this module.

Frame: [u32 magic='OFRP'][u16 opcode][u16 flags][u32 payload_len][payload].
All integers little-endian; strings are [u32 len][bytes].
"""

from __future__ import annotations

import socket
import struct

OFLD_RPC_MAGIC = 0x4F465250  # 'OFRP'

# OpCode values (src/common/protocol.h).
OP_GET_LATEST_SEALED_VERSION = 24
OP_GET_MANIFEST = 25
OP_PULL_TENSOR = 26

# TransportKind (rpc/offload_canonical.proto).
TK_TCP = 1
TK_FILE = 2
TK_LIBFABRIC_SEND = 3


class _W:
    def __init__(self):
        self.b = bytearray()

    def u8(self, v):  self.b += struct.pack("<B", v & 0xFF)
    def bool(self, v): self.b += struct.pack("<B", 1 if v else 0)
    def u16(self, v): self.b += struct.pack("<H", v & 0xFFFF)
    def u32(self, v): self.b += struct.pack("<I", v & 0xFFFFFFFF)
    def u64(self, v): self.b += struct.pack("<Q", v & 0xFFFFFFFFFFFFFFFF)
    def i32(self, v): self.b += struct.pack("<i", v)

    def str(self, s: str):
        data = s.encode("utf-8")
        self.u32(len(data))
        self.b += data


class _R:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0

    def _take(self, n):
        if self.p + n > len(self.d):
            raise ValueError("truncated wire message")
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def u8(self):   return self._take(1)[0]
    def bool(self): return self._take(1)[0] != 0
    def u16(self):  return struct.unpack("<H", self._take(2))[0]
    def u32(self):  return struct.unpack("<I", self._take(4))[0]
    def u64(self):  return struct.unpack("<Q", self._take(8))[0]
    def i32(self):  return struct.unpack("<i", self._take(4))[0]

    def str(self):
        n = self.u32()
        return self._take(n).decode("utf-8")


def make_frame(opcode: int, body: bytes, flags: int = 0) -> bytes:
    hdr = struct.pack("<IHHI", OFLD_RPC_MAGIC, opcode, flags, len(body))
    return hdr + body


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-frame")
        buf += chunk
    return bytes(buf)


def recv_frame(sock: socket.socket):
    """Return (opcode, payload_bytes)."""
    hdr = _recv_exact(sock, 12)
    magic, opcode, flags, plen = struct.unpack("<IHHI", hdr)
    if magic != OFLD_RPC_MAGIC:
        raise ValueError(f"bad frame magic {magic:#x}")
    payload = _recv_exact(sock, plen) if plen else b""
    return opcode, payload


# ---- JobKey helper ---------------------------------------------------------
def _put_jobkey(w: _W, tenant_id, job_id, launch_epoch, job_name):
    w.u64(tenant_id)
    w.u64(job_id)
    w.u64(launch_epoch)
    w.str(job_name or "")


def _get_jobkey(r: _R):
    return {
        "tenant_id": r.u64(),
        "job_id": r.u64(),
        "launch_epoch": r.u64(),
        "job_name": r.str(),
    }


def _get_key(r: _R) -> dict:
    job = _get_jobkey(r)
    k = {"job": job}
    k["model_role"] = r.u32()
    k["model_version"] = r.u64()
    k["param_id"] = r.u64()
    k["param_fqn_hash"] = r.u64()
    k["param_fqn_debug"] = r.str()
    k["dtype"] = r.u32()
    k["layout"] = r.u32()
    k["nbytes"] = r.u64()
    k["shape_hash"] = r.u64()
    k["stride_hash"] = r.u64()
    k["pp_rank"] = r.u32()
    k["tp_rank"] = r.u32()
    k["ep_rank"] = r.u32()
    k["etp_rank"] = r.u32()
    k["expert_id"] = r.i32()
    k["shard_offset"] = r.u64()
    k["shard_nbytes"] = r.u64()
    return k


# ---- GetLatestSealedVersion ------------------------------------------------
def encode_get_latest(tenant_id, job_id, launch_epoch, job_name, model_role) -> bytes:
    w = _W()
    _put_jobkey(w, tenant_id, job_id, launch_epoch, job_name)
    w.u32(model_role)
    return bytes(w.b)


def decode_get_latest_resp(payload: bytes) -> dict:
    r = _R(payload)
    return {"ok": r.bool(), "message": r.str(), "found": r.bool(),
            "model_version": r.u64()}


# ---- GetManifest -----------------------------------------------------------
def encode_get_manifest(tenant_id, job_id, launch_epoch, job_name, model_role,
                        model_version, flt=None) -> bytes:
    w = _W()
    _put_jobkey(w, tenant_id, job_id, launch_epoch, job_name)
    w.u32(model_role)
    w.u64(model_version)
    flt = flt or {}
    for name in ("pp", "tp", "ep", "etp"):
        present = name in flt
        w.bool(present)
        w.u32(int(flt.get(name, 0)))
    present = "expert" in flt
    w.bool(present)
    w.i32(int(flt.get("expert", 0)))
    return bytes(w.b)


def decode_get_manifest_resp(payload: bytes) -> dict:
    r = _R(payload)
    out = {"ok": r.bool(), "message": r.str(), "job": _get_jobkey(r),
           "model_role": r.u32(), "model_version": r.u64(), "state": r.u32()}
    n = r.u32()
    tensors = []
    for _ in range(n):
        key = _get_key(r)
        entry = {"key": key, "object_id": r.u64(), "nbytes": r.u64(),
                 "content_hash_lo": r.u64(), "content_hash_hi": r.u64(),
                 "location_hint": r.str()}
        tensors.append(entry)
    out["tensors"] = tensors
    return out


# ---- PullTensor ------------------------------------------------------------
def encode_pull(tenant_id, job_id, launch_epoch, job_name, model_role,
                model_version, object_id, transport, target_descriptor) -> bytes:
    w = _W()
    _put_jobkey(w, tenant_id, job_id, launch_epoch, job_name)
    w.u32(model_role)
    w.u64(model_version)
    w.u64(object_id)
    w.u32(transport)
    w.str(target_descriptor)
    return bytes(w.b)


def decode_pull_resp(payload: bytes) -> dict:
    r = _R(payload)
    return {"ok": r.bool(), "message": r.str(), "object_id": r.u64(),
            "nbytes": r.u64(), "transport": r.u32(),
            "content_hash_lo": r.u64(), "content_hash_hi": r.u64(),
            "transport_metadata": r.str()}


# ---- Export frame (OFEX) sent by the daemon's TCP/file transport ----------
EXPORT_MAGIC = 0x4F464558  # 'OFEX'
EXPORT_HEADER_FMT = "<IIQQQQ"   # magic, version, object_id, nbytes, hlo, hhi
EXPORT_HEADER_SIZE = struct.calcsize(EXPORT_HEADER_FMT)


def parse_export_header(hdr: bytes):
    magic, version, object_id, nbytes, hlo, hhi = struct.unpack(
        EXPORT_HEADER_FMT, hdr)
    if magic != EXPORT_MAGIC:
        raise ValueError(f"bad export frame magic {magic:#x}")
    return {"version": version, "object_id": object_id, "nbytes": nbytes,
            "content_hash_lo": hlo, "content_hash_hi": hhi}
