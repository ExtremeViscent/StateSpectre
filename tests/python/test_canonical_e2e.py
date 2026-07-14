"""End-to-end v2 canonical + rollout-pull test (needs a GPU + memfd daemon).

Usage: python test_canonical_e2e.py <unix_sock> <tcp_port>

Exercises the full v2 Python surface against a live daemon:
  - job-aware offload_context (register_job)
  - canonical_key builder (DP excluded, TP/expert included)
  - canonical_evict: create + DP-attach dedup (no second D2H)
  - seal + promote_rollout_version
  - RolloutWeightClient over the TCP control endpoint: discover latest sealed
    version, get manifest, diff, pull each tensor, verify bytes match the source.
"""

import sys
import torch

import state_spectre as ss
from state_spectre import _wire


def main():
    sock = sys.argv[1] if len(sys.argv) > 1 else "/tmp/state_spectre.sock"
    tcp_port = int(sys.argv[2]) if len(sys.argv) > 2 else 19099
    daemon_addr = f"unix://{sock}"
    dev = "cuda:0"

    # Build two source tensors (simulate 2 params of a rollout-visible version).
    torch.manual_seed(0)
    t1 = torch.randn(1024, 512, device=dev, dtype=torch.float32)
    t2 = torch.randn(2048, 256, device=dev, dtype=torch.float32)
    t1_ref = t1.clone().cpu()
    t2_ref = t2.clone().cpu()

    with ss.offload_context(daemon_addr=daemon_addr, device=dev, rank=0,
                            job_name="e2e_rollout_job",
                            scheduler_job_id=4242) as off:
        assert off.job_id is not None, "job registration failed"
        print(f"[trainer] job_id={off.job_id} launch_epoch={off.launch_epoch}")

        VER = 100
        k1 = off.canonical_key(model_role="policy_rollout", model_version=VER,
                               param_name="layers.0.mlp.w1", tensor=t1,
                               pp_rank=0, tp_rank=0, expert_id=-1)
        k2 = off.canonical_key(model_role="policy_rollout", model_version=VER,
                               param_name="layers.0.mlp.w2", tensor=t2,
                               pp_rank=0, tp_rank=0, expert_id=-1)

        h1 = off.canonical_evict(t1, k1, destructive=False, wait=True)
        h2 = off.canonical_evict(t2, k2, destructive=False, wait=True)
        print(f"[trainer] evict1 {h1.action_name} obj={h1.object_id} d2h={h1.did_d2h}")
        print(f"[trainer] evict2 {h2.action_name} obj={h2.object_id} d2h={h2.did_d2h}")
        assert h1.action_name == "NEED_D2H_CREATE" and h1.did_d2h
        assert h2.action_name == "NEED_D2H_CREATE" and h2.did_d2h

        # DP-equivalent re-evict of the SAME key must attach (no new D2H).
        t1b = t1_ref.to(dev)
        h1b = off.canonical_evict(t1b, k1, destructive=False, wait=True)
        print(f"[trainer] dp-attach {h1b.action_name} obj={h1b.object_id} d2h={h1b.did_d2h}")
        assert h1b.action_name == "ATTACHED_EXISTING", h1b.action_name
        assert h1b.object_id == h1.object_id
        assert not h1b.did_d2h, "dp-attach must skip D2H"

        # Seal + promote the rollout-visible version.
        res = off.promote_rollout_version(VER)
        print(f"[trainer] sealed: {res}")
        assert res["tensor_count"] == 2, res

        # Rollout side: pull over the TCP control endpoint.
        client = ss.RolloutWeightClient(
            daemon_addr=f"tcp://127.0.0.1:{tcp_port}",
            job_id=off.job_id, launch_epoch=off.launch_epoch,
            model_role="policy_rollout", recv_host="127.0.0.1")
        latest = client.get_latest_sealed_version()
        print(f"[rollout] latest sealed version = {latest}")
        assert latest == VER, latest
        manifest = client.get_manifest(latest)
        assert len(manifest["tensors"]) == 2, manifest
        changed = client.diff_local(manifest)
        assert len(changed) == 2, "all tensors changed on first pull"

        by_obj = {h1.object_id: t1_ref, h2.object_id: t2_ref}
        for e in manifest["tensors"]:
            raw = client.pull_tensor(e, latest, transport="tcp")
            got = torch.frombuffer(bytearray(raw), dtype=torch.float32)
            ref = by_obj[e["object_id"]].reshape(-1)
            assert got.numel() == ref.numel(), (got.numel(), ref.numel())
            assert torch.allclose(got, ref, atol=0, rtol=0), \
                f"byte mismatch for object {e['object_id']}"
            print(f"[rollout] pulled obj={e['object_id']} nbytes={e['nbytes']} OK")

        # Second diff must be empty (cache hit — nothing re-pulled).
        changed2 = client.diff_local(manifest)
        assert changed2 == [], f"expected empty diff after pull, got {changed2}"
        print("[rollout] second diff empty (cache works)")

    print("PASS: canonical + rollout pull e2e")


if __name__ == "__main__":
    main()
