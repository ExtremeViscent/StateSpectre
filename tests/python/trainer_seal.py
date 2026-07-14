"""Trainer side for the libfabric cross-node pull test (torch-base env, 3.13).

Registers a job (deterministic job_id = scheduler_job_id), canonical-evicts two
tensors as POLICY_ROLLOUT@version, seals+promotes, writes the source bytes to
<workdir> for the receiver to verify against, signals 'ready', and holds the
offload context open until it sees 'done' (so the canonical objects stay alive
while the rollout side pulls them).

Usage: python trainer_seal.py <unix_sock> <workdir> <scheduler_job_id> <version>
"""

import os
import sys
import time
import torch

import state_spectre as ss

SCHED = None


def main():
    sock = sys.argv[1]
    workdir = sys.argv[2]
    sched_id = int(sys.argv[3])
    version = int(sys.argv[4])
    os.makedirs(workdir, exist_ok=True)

    torch.manual_seed(sched_id)
    specs = [("layers.0.mlp.w1", (1024, 512)), ("layers.0.mlp.w2", (2048, 256))]
    tensors = {name: torch.randn(*shape, device="cuda:0", dtype=torch.float32)
               for name, shape in specs}

    with ss.offload_context(daemon_addr=f"unix://{sock}", device="cuda:0", rank=0,
                            job_name="lf_xnode_job", scheduler_job_id=sched_id) as off:
        print(f"[trainer] job_id={off.job_id} launch_epoch={off.launch_epoch}",
              flush=True)
        obj_ids = {}
        for name, _ in specs:
            t = tensors[name]
            key = off.canonical_key(model_role="policy_rollout",
                                    model_version=version, param_name=name,
                                    tensor=t, pp_rank=0, tp_rank=0, expert_id=-1)
            h = off.canonical_evict(t, key, destructive=False, wait=True)
            obj_ids[name] = h.object_id
            # Persist source bytes for the receiver to verify.
            with open(os.path.join(workdir, f"obj_{h.object_id}.bin"), "wb") as f:
                f.write(t.detach().cpu().contiguous().numpy().tobytes())
            print(f"[trainer] evict {name} -> obj={h.object_id} "
                  f"action={h.action_name}", flush=True)

        res = off.promote_rollout_version(version)
        print(f"[trainer] sealed+promoted: {res}", flush=True)

        # Record identity for the receiver.
        with open(os.path.join(workdir, "job.txt"), "w") as f:
            f.write(f"{off.job_id} {off.launch_epoch} {version}\n")
        with open(os.path.join(workdir, "ready"), "w") as f:
            f.write("ready\n")

        # Hold objects alive until the receiver signals completion (or timeout).
        deadline = time.time() + 120
        while time.time() < deadline:
            if os.path.exists(os.path.join(workdir, "done")):
                break
            time.sleep(0.5)
        print("[trainer] done", flush=True)


if __name__ == "__main__":
    main()
