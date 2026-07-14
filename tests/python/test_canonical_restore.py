"""Canonical offload/reload round-trip test (needs a GPU + a running daemon).

Reproduces the NexTrainer offload_actor()/load_actor() symmetric round-trip with
DP dedup: rank0 creates a canonical object (real D2H), a DP replica evicts the
SAME key and gets ATTACHED_EXISTING (no D2H), and BOTH read the one shared copy
back into GPU storage by object_id via CanonicalHandle.restore_into — byte-exact.
This is the path that was missing (the API was publish-only before).

Usage: python test_canonical_restore.py <unix_sock>
"""

import sys
import torch
import state_spectre as ss


def main():
    sock = sys.argv[1] if len(sys.argv) > 1 else "/tmp/state_spectre.sock"
    dev = "cuda:0"

    with ss.offload_context(daemon_addr=f"unix://{sock}", device=dev, rank=0,
                            job_name="reload_job", scheduler_job_id=321) as off:
        torch.manual_seed(1)
        w = torch.randn(2048, 512, device=dev, dtype=torch.float32)
        ref = w.clone().cpu()
        nbytes = w.numel() * w.element_size()

        key = off.canonical_key(model_role="policy_trainable", model_version=7,
                                param_name="layers.3.attn.wqkv", tensor=w,
                                pp_rank=0, tp_rank=0, expert_id=-1)

        # rank0-style create: real D2H into the canonical object.
        h = off.canonical_evict(w, key, destructive=False, wait=True)
        assert h.action_name == "NEED_D2H_CREATE" and h.did_d2h, h.action_name
        print(f"[create] object_id={h.object_id} action={h.action_name}", flush=True)

        # Reload the SAME params back by object_id (offload_actor -> load_actor).
        out = torch.empty_like(w)
        h.restore_into(out, wait=True)
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), ref), "creator reload mismatch"
        print("[reload] creator restore_into byte-exact", flush=True)

        # DP-replica path: same key -> ATTACHED_EXISTING, NO local D2H, yet the
        # replica can read the shared bytes back by object_id.
        w2 = ref.to(dev)                       # an identical replica of the shard
        h2 = off.canonical_evict(w2, key, destructive=False, wait=True)
        assert h2.action_name == "ATTACHED_EXISTING", h2.action_name
        assert not h2.did_d2h, "replica must not D2H"
        assert h2.object_id == h.object_id
        out2 = torch.empty_like(w)
        h2.restore_into(out2, wait=True)       # replica reloads the shared copy
        torch.cuda.synchronize()
        assert torch.equal(out2.cpu(), ref), "attached-replica reload mismatch"
        print("[reload] ATTACHED replica (no D2H) restore_into byte-exact", flush=True)

        # Aliased flat-buffer variant: free VRAM via storage resize, then reload
        # into the SAME storage by object_id.
        buf = ref.reshape(-1).to(dev)
        kbuf = off.canonical_key(model_role="policy_trainable", model_version=7,
                                 param_name="flat.buffer", tensor=buf)
        hb = off.canonical_evict(buf, kbuf, destructive=False, wait=True)
        buf.untyped_storage().resize_(0)
        assert buf.untyped_storage().size() == 0
        buf.untyped_storage().resize_(nbytes)
        hb.restore_into(buf, wait=True)
        torch.cuda.synchronize()
        assert torch.equal(buf.cpu(), ref.reshape(-1)), "flat-buffer reload mismatch"
        print("[reload] aliased flat-buffer resize+restore_into byte-exact", flush=True)

    print("PASS: canonical offload/reload round-trip (create, attached-replica, flat-buffer)")


if __name__ == "__main__":
    main()
