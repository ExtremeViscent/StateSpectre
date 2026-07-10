"""RolloutWeightClient — remote weight-pull client for rollout/inference engines.

Mirrors the NorthCheckpoint pull pattern (discover version -> get manifest ->
diff local cache -> pull changed tensors) but layered over the offload manager's
sealed-manifest control plane, so a rollout engine pulls ONLY sealed model
versions, never mutable trainer weights (03_ROLLOUT_PULL_AND_EXPORT.md).

Transport: the daemon is a *request-triggered sender*. For the debug TCP
transport this client opens a receiver socket, asks the daemon to
PullTensor(TCP, "<our_host>:<port>"), and reads the OFEX frame the daemon sends.
For libfabric the target descriptor is "<nic>|<ib_address_hex>" (the client's
north_comm listener address); the daemon connects and sends via north_comm.

Local cache: keyed by (job_id, launch_epoch, model_role, model_version,
object_id, content_hash) so unchanged tensors across versions are not re-pulled.
"""

from __future__ import annotations

import socket
import struct
from typing import Optional

from . import _wire
from ._wire import (TK_TCP, TK_LIBFABRIC_SEND, OP_GET_LATEST_SEALED_VERSION,
                    OP_GET_MANIFEST, OP_PULL_TENSOR)


class RolloutWeightClient:
    """Pull sealed model weights from a trainer-side offload daemon.

    Parameters
    ----------
    daemon_addr : "tcp://host:port"  (the daemon's v2 control TCP endpoint)
    job_id, launch_epoch : job identity (from the trainer's off.job_id / epoch)
    model_role : "policy_rollout" by default (only sealed roles are pullable)
    tenant_id : tenant scope (default 0)
    """

    def __init__(self, daemon_addr: str, job_id: int, launch_epoch: int,
                 model_role="policy_rollout", tenant_id: int = 0,
                 job_name: str = "", recv_host: Optional[str] = None):
        if not daemon_addr.startswith("tcp://"):
            raise ValueError("daemon_addr must be tcp://host:port for rollout pull")
        hostport = daemon_addr[len("tcp://"):]
        host, _, port = hostport.rpartition(":")
        self._daemon_host = host
        self._daemon_port = int(port)
        self.tenant_id = int(tenant_id)
        self.job_id = int(job_id)
        self.launch_epoch = int(launch_epoch)
        self.job_name = job_name
        # Lazy import to avoid a hard torch dep at module import for pure control.
        from . import ModelRole
        self.model_role = ModelRole.parse(model_role)
        # host the daemon should connect back to for TCP transport.
        self._recv_host = recv_host or socket.gethostbyname(socket.gethostname())
        # cache: object_id -> (content_hash_lo, content_hash_hi)
        self._cache = {}

    # ---- control-plane round trip ---------------------------------------- #
    def _rpc(self, opcode: int, body: bytes):
        with socket.create_connection((self._daemon_host, self._daemon_port),
                                      timeout=30) as s:
            s.sendall(_wire.make_frame(opcode, body))
            rop, payload = _wire.recv_frame(s)
            if rop != opcode:
                raise IOError(f"reply opcode mismatch: sent {opcode} got {rop}")
            return payload

    def get_latest_sealed_version(self) -> Optional[int]:
        body = _wire.encode_get_latest(self.tenant_id, self.job_id,
                                       self.launch_epoch, self.job_name,
                                       self.model_role)
        resp = _wire.decode_get_latest_resp(self._rpc(OP_GET_LATEST_SEALED_VERSION, body))
        if not resp["ok"] or not resp["found"]:
            return None
        return resp["model_version"]

    def get_manifest(self, model_version: int, flt: dict = None) -> dict:
        body = _wire.encode_get_manifest(self.tenant_id, self.job_id,
                                         self.launch_epoch, self.job_name,
                                         self.model_role, int(model_version), flt)
        resp = _wire.decode_get_manifest_resp(self._rpc(OP_GET_MANIFEST, body))
        if not resp["ok"]:
            raise IOError(f"get_manifest failed: {resp['message']}")
        return resp

    def diff_local(self, manifest: dict) -> list:
        """Return manifest entries whose (object_id, content_hash) is not cached."""
        changed = []
        for e in manifest["tensors"]:
            key = e["object_id"]
            cached = self._cache.get(key)
            cur = (e["content_hash_lo"], e["content_hash_hi"])
            if cached != cur:
                changed.append(e)
        return changed

    # ---- data-plane pull -------------------------------------------------- #
    def pull_tensor(self, entry: dict, model_version: int,
                    transport: str = "tcp",
                    libfabric_descriptor: Optional[str] = None) -> bytes:
        """Pull one manifest entry's bytes. Returns the raw tensor bytes.

        For transport="tcp" (debug), this opens a one-shot receiver, asks the
        daemon to send, reads the OFEX frame, verifies object_id/nbytes, updates
        the local cache, and returns the payload bytes. For transport="libfabric"
        the caller must run a north_comm listener and pass its descriptor; this
        method issues the PullTensor and returns b"" (bytes land in the caller's
        registered buffer via north_comm).
        """
        object_id = int(entry["object_id"])
        expect_nbytes = int(entry["nbytes"])

        if transport == "tcp":
            # Stand up a one-shot receiver the daemon will connect back to.
            lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            lsock.bind(("0.0.0.0", 0))
            lsock.listen(1)
            port = lsock.getsockname()[1]
            target = f"{self._recv_host}:{port}"
            body = _wire.encode_pull(self.tenant_id, self.job_id,
                                     self.launch_epoch, self.job_name,
                                     self.model_role, int(model_version),
                                     object_id, TK_TCP, target)
            # Fire the pull from a background thread so we can accept the daemon's
            # connection (the daemon connects to us as part of handling PullTensor).
            import threading
            result = {}

            def _do_rpc():
                try:
                    result["resp"] = _wire.decode_pull_resp(
                        self._rpc(OP_PULL_TENSOR, body))
                except Exception as ex:  # noqa: BLE001
                    result["err"] = ex

            th = threading.Thread(target=_do_rpc, daemon=True)
            th.start()
            lsock.settimeout(30)
            conn, _ = lsock.accept()
            with conn:
                hdr = _recv_exact(conn, _wire.EXPORT_HEADER_SIZE)
                meta = _wire.parse_export_header(hdr)
                payload = _recv_exact(conn, meta["nbytes"])
            lsock.close()
            th.join(timeout=30)
            if "err" in result:
                raise result["err"]
            resp = result.get("resp", {})
            if not resp.get("ok", False):
                raise IOError(f"pull_tensor failed: {resp.get('message')}")
            if meta["object_id"] != object_id or meta["nbytes"] != expect_nbytes:
                raise IOError("pulled object/nbytes mismatch vs manifest")
            self._cache[object_id] = (entry["content_hash_lo"],
                                      entry["content_hash_hi"])
            return payload

        elif transport in ("libfabric", "libfabric_send"):
            if not libfabric_descriptor:
                raise ValueError("libfabric transport needs libfabric_descriptor "
                                 "('<nic>|<ib_address_hex>' of the client listener)")
            body = _wire.encode_pull(self.tenant_id, self.job_id,
                                     self.launch_epoch, self.job_name,
                                     self.model_role, int(model_version),
                                     object_id, TK_LIBFABRIC_SEND,
                                     libfabric_descriptor)
            resp = _wire.decode_pull_resp(self._rpc(OP_PULL_TENSOR, body))
            if not resp.get("ok", False):
                raise IOError(f"pull_tensor(libfabric) failed: {resp.get('message')}")
            self._cache[object_id] = (entry["content_hash_lo"],
                                      entry["content_hash_hi"])
            return b""  # bytes delivered into the caller's north_comm buffer

        raise ValueError(f"unknown transport {transport!r}")

    def sync_latest(self, transport: str = "tcp") -> dict:
        """Convenience: discover latest sealed version, pull all changed tensors.

        Returns {"version": v, "pulled": [object_id...], "skipped": [object_id...]}.
        """
        version = self.get_latest_sealed_version()
        if version is None:
            return {"version": None, "pulled": [], "skipped": []}
        manifest = self.get_manifest(version)
        changed = self.diff_local(manifest)
        pulled, skipped = [], []
        changed_ids = {e["object_id"] for e in changed}
        for e in manifest["tensors"]:
            if e["object_id"] in changed_ids:
                self.pull_tensor(e, version, transport=transport)
                pulled.append(e["object_id"])
            else:
                skipped.append(e["object_id"])
        return {"version": version, "pulled": pulled, "skipped": skipped}


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-payload")
        buf += chunk
    return bytes(buf)
