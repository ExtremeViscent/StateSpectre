"""Rollout-side libfabric receiver for cross-node PullTensor validation.

Runs under the `bprl` conda env (Python 3.12) where `north_comm` imports.
Pure-Python: imports state_spectre._wire directly by path (no torch / no
_state_spectre C extension needed on this side).

Flow (rollout engine side):
  1. stand up a north_comm listener on the given NIC; capture its IB address.
  2. control-plane: GetLatestSealedVersion + GetManifest over the daemon's TCP
     port to learn which objects to pull.
  3. for each object: post a north_comm recv, then issue PullTensor(libfabric,
     "<nic>|<ib_addr>") to the daemon; the daemon connects back and sends the
     object bytes via libfabric RDMA. Verify nbytes + optional content bytes.

Usage:
  python rollout_libfabric_recv.py <daemon_host> <tcp_port> <job_id> \
         <launch_epoch> <nic> [expected_sha_dir]
"""

import importlib.util
import os
import socket
import sys
import threading

# Import state_spectre._wire by file path (avoids the package __init__ which pulls
# the torch/_state_spectre C extension not available in this env).
_HERE = os.path.dirname(os.path.abspath(__file__))
_WIRE_PATH = os.path.join(_HERE, os.pardir, os.pardir, "python_api",
                          "state_spectre", "_wire.py")
_spec = importlib.util.spec_from_file_location("_ofld_wire", _WIRE_PATH)
_wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wire)

import north_comm  # noqa: E402  (bprl env)

MODEL_ROLE_POLICY_ROLLOUT = 2


def rpc(host, port, opcode, body):
    with socket.create_connection((host, port), timeout=30) as s:
        s.sendall(_wire.make_frame(opcode, body))
        rop, payload = _wire.recv_frame(s)
        assert rop == opcode, (rop, opcode)
        return payload


def main():
    daemon_host = sys.argv[1]
    tcp_port = int(sys.argv[2])
    job_id = int(sys.argv[3])
    launch_epoch = int(sys.argv[4])
    nic = sys.argv[5]
    workdir = sys.argv[6] if len(sys.argv) > 6 else None  # for content verify

    # 1. north_comm listener. Its callback receives one Buffer per accepted
    #    connection (the daemon opens one connection per PullTensor send).
    received = []
    got_evt = threading.Event()

    def on_conn(ep: "north_comm.Endpoint"):
        # The daemon sends each object as a 1-D uint8 Tensor buffer; recover it
        # losslessly via DLPack -> torch -> raw bytes.
        import torch
        buf = ep.recv(north_comm.Device.cpu())
        t = torch.from_dlpack(buf).contiguous().view(torch.uint8)
        received.append(bytes(t.numpy().tobytes()))
        got_evt.set()

    listener = north_comm.create_listener(nic, on_conn, north_comm.Device.cpu())
    listener.start()
    ib_addr = listener.get_address()
    target = f"{nic}|{ib_addr}"
    print(f"[rollout] listener up nic={nic} ib_addr={ib_addr}", flush=True)

    # 2. discover latest sealed version + manifest.
    body = _wire.encode_get_latest(0, job_id, launch_epoch, "",
                                   MODEL_ROLE_POLICY_ROLLOUT)
    r = _wire.decode_get_latest_resp(rpc(daemon_host, tcp_port,
                                         _wire.OP_GET_LATEST_SEALED_VERSION, body))
    assert r["ok"] and r["found"], r
    version = r["model_version"]
    print(f"[rollout] latest sealed version = {version}", flush=True)

    body = _wire.encode_get_manifest(0, job_id, launch_epoch, "",
                                     MODEL_ROLE_POLICY_ROLLOUT, version)
    man = _wire.decode_get_manifest_resp(rpc(daemon_host, tcp_port,
                                             _wire.OP_GET_MANIFEST, body))
    assert man["ok"], man
    print(f"[rollout] manifest: {len(man['tensors'])} tensors", flush=True)

    # 3. pull each object over libfabric.
    pulled = 0
    for e in man["tensors"]:
        got_evt.clear()
        before = len(received)
        pbody = _wire.encode_pull(0, job_id, launch_epoch, "",
                                  MODEL_ROLE_POLICY_ROLLOUT, version,
                                  e["object_id"], _wire.TK_LIBFABRIC_SEND, target)
        presp = _wire.decode_pull_resp(rpc(daemon_host, tcp_port,
                                           _wire.OP_PULL_TENSOR, pbody))
        if not presp["ok"]:
            print(f"[rollout] PULL FAIL obj={e['object_id']}: {presp['message']}",
                  flush=True)
            listener.stop()
            sys.exit(1)
        # wait for the north_comm recv callback to land the bytes.
        if not got_evt.wait(timeout=30):
            print(f"[rollout] TIMEOUT waiting recv obj={e['object_id']}", flush=True)
            listener.stop()
            sys.exit(2)
        payload = received[before]           # raw bytes
        nbytes = len(payload)
        ok = (nbytes == e["nbytes"])
        # Optional byte-exact content check against the trainer's source dump.
        content_ok = True
        if ok and workdir:
            src_path = os.path.join(workdir, f"obj_{e['object_id']}.bin")
            if os.path.exists(src_path):
                with open(src_path, "rb") as f:
                    src = f.read()
                content_ok = (payload == src)
        verdict = "OK" if (ok and content_ok) else "MISMATCH"
        print(f"[rollout] pulled obj={e['object_id']} nbytes={nbytes} "
              f"expect={e['nbytes']} content={'ok' if content_ok else 'BAD'} "
              f"{verdict}", flush=True)
        if not (ok and content_ok):
            listener.stop()
            sys.exit(3)
        pulled += 1

    listener.stop()
    print(f"[rollout] PASS: pulled {pulled} objects over libfabric", flush=True)


if __name__ == "__main__":
    main()
