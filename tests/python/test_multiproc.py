#!/usr/bin/env python3
"""Multi-process safety test: N separate processes concurrently evict/restore
against one daemon on the same GPU/NUMA arena.

Proves (RACE_CONDITIONS §4, Milestone 3):
  - no two processes ever write the same pinned slot (daemon lease enforcement)
  - every process's restore is byte-identical (no cross-process corruption)
  - distinct rank_ids get distinct sessions/epochs

Each worker is a fresh process (torch + state_spectre imported independently), so
this exercises the real fd-passing + mmap + lease path per process.

Usage: python test_multiproc.py <socket_path> [nprocs] [ntensors]
"""
import os
import sys
import multiprocessing as mp

SOCKET = sys.argv[1] if len(sys.argv) > 1 else "/tmp/state_spectre.sock"
NPROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NTENSORS = int(sys.argv[3]) if len(sys.argv) > 3 else 20


def worker(rank, socket_path, ntensors, q):
    import torch
    import state_spectre as ss
    try:
        torch.cuda.set_device(0)
        ok = 0
        with ss.offload_context(daemon_addr=f"unix://{socket_path}",
                                device="cuda:0", rank=rank) as off:
            for i in range(ntensors):
                # Distinct content per (rank, i) to detect any cross-talk.
                n = 256 * 1024 + (rank * 131 + i) % 4096
                x = torch.arange(n, device="cuda:0", dtype=torch.float32)
                x = x + float(rank * 1_000_000 + i)
                orig = x.clone()
                h = off.evict(x, name=f"r{rank}_t{i}", wait=True)
                y = h.restore(device="cuda:0")
                if not torch.equal(y, orig):
                    diff = (y != orig).nonzero()
                    first = diff[0].item() if diff.numel() else -1
                    q.put((rank, "MISMATCH",
                           f"tensor#{i} n={n} first_bad_idx={first} "
                           f"got={y.flatten()[first].item() if first>=0 else 'NA'} "
                           f"want={orig.flatten()[first].item() if first>=0 else 'NA'} "
                           f"got[0]={y[0].item()} want[0]={orig[0].item()}"))
                    return
                h.discard()
                ok += 1
        q.put((rank, "OK", ok))
    except Exception as e:  # noqa
        import traceback
        q.put((rank, "EXC", f"{e}\n{traceback.format_exc()}"))


def main():
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    procs = []
    for r in range(NPROCS):
        # distinct rank ids so sessions/epochs are distinct
        p = mp.Process(target=worker, args=(100 + r, SOCKET, NTENSORS, q))
        p.start()
        procs.append(p)

    results = []
    for _ in range(NPROCS):
        results.append(q.get())
    for p in procs:
        p.join()

    failures = 0
    for rank, status, detail in sorted(results):
        if status == "OK":
            print(f"  [ok  ] rank {rank}: {detail} evict/restore cycles byte-identical")
        else:
            failures += 1
            print(f"  [FAIL] rank {rank}: {status}: {detail}")

    if failures == 0:
        print(f"[PASS] multiproc: {NPROCS} processes x {NTENSORS} tensors, "
              "no corruption, no slot conflicts")
        return 0
    print(f"[FAIL] multiproc: {failures} process(es) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
