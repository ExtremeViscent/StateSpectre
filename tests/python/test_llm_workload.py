#!/usr/bin/env python3
"""Representative LLM-style integration test for the fastoffload runtime.

This simulates the memory-offload pattern of transformer inference/training:
per-layer weights and a growing KV cache are offloaded from VRAM to host/NVMe
tiers and restored on demand, verifying both *correctness* (byte-identical
restores) and *real VRAM release* (torch.cuda.memory_allocated drops after a
destructive evict + wait, per DESIGN_SPEC §10).

Style mirrors tests/python/test_python_e2e.py:
  - prints "[ok  ]" / "[FAIL]" per check, counts failures
  - returns 0 on success, nonzero if any check failed
  - takes the daemon socket path as sys.argv[1]

It can also launch its own daemon:
  python test_llm_workload.py --spawn
  python test_llm_workload.py /tmp/some.sock            # connect to existing

Sizes are overridable via env:
  OFLD_LLM_LAYER_MB   (default 128)   per-layer weight size, MiB, bf16
  OFLD_LLM_LAYERS     (default 8)     number of transformer layers L
  OFLD_LLM_KV_STEPS   (default 16)    number of simulated decode steps
"""
import math
import os
import subprocess
import sys
import tempfile
import time

import torch

import fastoffload as fo


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
LAYER_MB = int(os.environ.get("OFLD_LLM_LAYER_MB", "128"))
LAYERS = int(os.environ.get("OFLD_LLM_LAYERS", "8"))
KV_STEPS = int(os.environ.get("OFLD_LLM_KV_STEPS", "16"))
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
ELEM = 2  # bytes per bf16 element


def _square_dim(mb: int) -> int:
    """Side length of a square bf16 matrix that occupies ~`mb` MiB."""
    elems = (mb * (1 << 20)) // ELEM
    return int(math.isqrt(elems))


def mb(nbytes: int) -> float:
    return nbytes / (1 << 20)


# Tiers that mean "no longer occupying a pinned staging slot" — i.e. the
# daemon's background drain has migrated the bytes to host DRAM / NVMe.
_COLD_TIERS = ("pageable", "nvme")


def wait_drained(handle, timeout_s: float = 15.0) -> str:
    """Block until an evicted tensor has drained out of the pinned staging tier.

    The pinned arena is a scarce, shared staging area. After a destructive
    ``evict`` the D2H copy lands in ``pinned``; the daemon then asynchronously
    drains it to a cold tier (``pageable``/``nvme``), which is what actually
    frees the pinned slots for reuse. Restoring while still ``pinned`` keeps the
    slot occupied and, under the layer-ahead prefetch pattern, quickly exhausts
    the arena. Waiting for the cold tier mirrors the real LLM offload pattern:
    park cold layers / KV history in host DRAM or NVMe, read them back on
    demand. Returns the final location string.
    """
    deadline = time.time() + timeout_s
    loc = handle.location()
    while loc not in _COLD_TIERS and time.time() < deadline:
        time.sleep(0.005)
        loc = handle.location()
    return loc



# --------------------------------------------------------------------------- #
# daemon spawning (optional --spawn)
# --------------------------------------------------------------------------- #
def _find_daemon() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    cand = os.path.join(root, "build", "offloadd")
    if not os.path.isfile(cand):
        raise FileNotFoundError(f"daemon binary not found at {cand}; build it first")
    return cand


def _spawn_daemon():
    """Start build/offloadd on a fresh socket; return (proc, socket_path)."""
    binary = _find_daemon()
    sock = tempfile.mktemp(prefix="fastoffload_llm_", suffix=".sock")
    if os.path.exists(sock):
        os.unlink(sock)
    env = dict(os.environ)
    env.setdefault("OFLD_LOG_LEVEL", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Arena large enough to hold all offloaded layers + KV comfortably.
    arena_mb = max(8192, LAYER_MB * (LAYERS + 4))
    proc = subprocess.Popen(
        [binary, "--smoke-arena-mb", str(arena_mb), "--numa", "0",
         "--gpu", "0", "--socket", sock],
        env=env,
    )
    # Wait for the socket to appear (daemon ready), up to a few seconds.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if os.path.exists(sock):
            time.sleep(0.5)  # small grace for the accept loop
            return proc, sock
        if proc.poll() is not None:
            raise RuntimeError(
                f"daemon exited early with code {proc.returncode}")
        time.sleep(0.1)
    proc.terminate()
    raise TimeoutError(f"daemon did not create socket {sock} in time")


# --------------------------------------------------------------------------- #
# main workload
# --------------------------------------------------------------------------- #
def run(socket_path: str) -> int:
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.manual_seed(1234)

    failures = 0

    def check(cond, msg):
        nonlocal failures
        status = "ok  " if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {msg}")

    dim = _square_dim(LAYER_MB)
    layer_bytes = dim * dim * ELEM
    print(f"config: L={LAYERS} layers, dim={dim} ({mb(layer_bytes):.1f} MiB/layer, "
          f"bf16), KV_STEPS={KV_STEPS}, total weights "
          f"{mb(layer_bytes * LAYERS):.1f} MiB")

    with fo.offload_context(
        daemon_addr=f"unix://{socket_path}",
        device=DEVICE,
        rank=0,
        invalidate_mode="set_empty",
    ) as off:
        stream = torch.cuda.current_stream()

        # ------------------------------------------------------------------ #
        # Build the "model": L layer weights on GPU + a pristine CPU copy of
        # each (host reference for byte-identical verification on restore).
        # ------------------------------------------------------------------ #
        print("\nsetup: materialize model weights on GPU + save CPU references")
        weights = []
        cpu_ref = []
        for i in range(LAYERS):
            w = torch.randn(dim, dim, device=DEVICE, dtype=DTYPE)
            weights.append(w)
            cpu_ref.append(w.detach().to("cpu").clone())
        torch.cuda.synchronize()
        check(len(weights) == LAYERS, f"created {LAYERS} layer weights on GPU")

        # ================================================================== #
        # PHASE A: offload cold layers, prove VRAM is really released.
        # Keep layer `active` resident; evict all the others.
        # ================================================================== #
        print("\nPhase A: evict cold layers and verify real VRAM release")
        active = 0
        alloc_before = torch.cuda.memory_allocated()

        handles = {}  # layer index -> OffloadHandle
        for i in range(LAYERS):
            if i == active:
                continue
            handles[i] = off.evict(weights[i], name=f"layer{i}.weight")
            weights[i] = None  # drop python ref to the (now-invalid) storage
        off.wait(list(handles.values()))
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated()

        n_evicted = len(handles)
        evicted_volume = n_evicted * layer_bytes
        dropped = alloc_before - alloc_after
        print(f"  memory_allocated before evict: {mb(alloc_before):.1f} MiB")
        print(f"  memory_allocated after  evict: {mb(alloc_after):.1f} MiB")
        print(f"  evicted {n_evicted} layers = {mb(evicted_volume):.1f} MiB; "
              f"VRAM dropped by {mb(dropped):.1f} MiB "
              f"({100.0 * dropped / evicted_volume:.1f}% of evicted volume)")
        check(dropped >= 0.5 * evicted_volume,
              f"VRAM allocated dropped by >= 50% of evicted volume "
              f"({mb(dropped):.1f} >= {mb(0.5 * evicted_volume):.1f} MiB)")
        # Let the cold layers drain out of the pinned staging tier into host
        # DRAM / NVMe. This both frees pinned slots for the read-back path and
        # models parking cold weights in cheap memory between uses.
        for i in handles:
            loc = wait_drained(handles[i])
            check(loc != "gpu",
                  f"layer{i} no longer resident on GPU (location={loc})")

        # ================================================================== #
        # PHASE B: forward pass with prefetch/compute overlap.
        # For each layer: restore it (prefetched when possible), run a matmul
        # to simulate compute while the *next* layer is prefetched, and verify
        # the restored weights are byte-identical to the saved reference.
        # ================================================================== #
        print("\nPhase B: forward pass with prefetch overlap + byte-identical check")
        act = torch.randn(dim, 8, device=DEVICE, dtype=DTYPE)
        identical = 0
        for i in range(LAYERS):
            if i == active:
                w = weights[i]  # never evicted, still resident
            else:
                # restore() consumes the outstanding prefetch (if any) instead
                # of issuing a duplicate H2D.
                w = handles[i].restore(device=DEVICE, stream=stream)

            # Kick off prefetch of the next evicted layer so its H2D overlaps
            # with this layer's compute (classic layer-ahead prefetch).
            nxt = i + 1
            if nxt < LAYERS and nxt != active and nxt in handles:
                handles[nxt].prefetch(device=DEVICE, stream=stream)

            # "Compute" on this layer (simulated attention/MLP matmul).
            _ = torch.matmul(w, act)

            # Correctness: restored weights must match the pristine reference.
            same = torch.equal(w.detach().to("cpu"), cpu_ref[i])
            if same:
                identical += 1
            check(same, f"layer{i} restored weight byte-identical to reference")

            # Resource hygiene: once a layer's weights are back on-GPU and used,
            # release its offload lease so the pinned/cold slots are reclaimed
            # (mirrors freeing a layer after its forward step). Keeps the pinned
            # arena footprint bounded to ~2 layers during the sweep.
            if i != active:
                handles[i].discard()
                handles[i] = None

        check(identical == LAYERS,
              f"all {LAYERS} layers byte-identical after forward pass "
              f"({identical}/{LAYERS})")
        torch.cuda.synchronize()

        # ================================================================== #
        # PHASE C: KV-cache grow / offload / restore across decode steps.
        # A preallocated cache is filled one step at a time; every few steps
        # the whole cache is evicted then restored, verifying its contents
        # survive the round trip.
        # ================================================================== #
        print("\nPhase C: KV-cache grow / evict / restore across decode steps")
        kv_dim = dim  # per-step key/value width
        kv = torch.zeros(KV_STEPS, kv_dim, device=DEVICE, dtype=DTYPE)
        kv_ref = torch.zeros(KV_STEPS, kv_dim, dtype=DTYPE)  # host mirror
        kv_roundtrips = 0
        kv_ok = 0
        for step in range(KV_STEPS):
            row = torch.randn(kv_dim, device=DEVICE, dtype=DTYPE)
            kv[step] = row
            kv_ref[step] = row.detach().to("cpu")

            # Periodically evict + restore the cache (e.g. to free VRAM between
            # bursts) and verify nothing was corrupted.
            if (step + 1) % 4 == 0:
                snapshot = kv.detach().to("cpu").clone()
                hkv = off.evict(kv, name=f"kv@step{step}", wait=True)
                loc = wait_drained(hkv)
                check(loc != "gpu",
                      f"KV cache off-GPU after evict at step {step} "
                      f"(location={loc})")
                kv = hkv.restore(device=DEVICE, stream=stream)
                kv_roundtrips += 1
                ok = torch.equal(kv.detach().to("cpu"), snapshot)
                # Also confirm it matches the independent host mirror.
                ok = ok and torch.equal(kv.detach().to("cpu"), kv_ref)
                if ok:
                    kv_ok += 1
                check(ok, f"KV cache contents intact after roundtrip "
                          f"#{kv_roundtrips} (step {step})")
                # Release the offload lease for this cache snapshot now that it
                # is back on-GPU (reclaim pinned/cold slots for the next burst).
                hkv.discard()
        check(kv_roundtrips > 0 and kv_ok == kv_roundtrips,
              f"all {kv_roundtrips} KV-cache roundtrips verified")
        torch.cuda.synchronize()

        # ================================================================== #
        # PHASE D: batch evict_many / restore_many for a group of layers,
        # verify every tensor is byte-identical in one coalesced call.
        # ================================================================== #
        print("\nPhase D: evict_many / restore_many batch, byte-identical check")
        n_batch = min(4, LAYERS)
        batch = []
        batch_ref = []
        for j in range(n_batch):
            t = torch.randn(dim, dim, device=DEVICE, dtype=DTYPE)
            batch.append(t)
            batch_ref.append(t.detach().to("cpu").clone())
        batch_names = [f"batch{j}" for j in range(n_batch)]

        alloc_pre_batch = torch.cuda.memory_allocated()
        bhandles = off.evict_many(batch, names=batch_names, stream=stream)
        off.wait(bhandles)
        torch.cuda.synchronize()
        alloc_post_batch = torch.cuda.memory_allocated()
        # Drop our python refs to the invalidated originals.
        for j in range(n_batch):
            batch[j] = None
        batch_dropped = alloc_pre_batch - alloc_post_batch
        print(f"  evict_many freed {mb(batch_dropped):.1f} MiB "
              f"({n_batch} x {mb(layer_bytes):.1f} MiB)")
        check(batch_dropped >= 0.5 * n_batch * layer_bytes,
              f"evict_many released >= 50% of batch volume "
              f"({mb(batch_dropped):.1f} MiB)")

        # Drain the batch to the cold tier before reading it back.
        for h in bhandles:
            wait_drained(h)
        restored = off.restore_many(bhandles, device=DEVICE, stream=stream)
        torch.cuda.synchronize()
        check(len(restored) == n_batch,
              f"restore_many returned {n_batch} tensors")
        batch_identical = 0
        for j in range(n_batch):
            same = torch.equal(restored[j].detach().to("cpu"), batch_ref[j])
            if same:
                batch_identical += 1
            check(same, f"batch tensor {j} byte-identical after restore_many")
        check(batch_identical == n_batch,
              f"all {n_batch} batched tensors byte-identical "
              f"({batch_identical}/{n_batch})")

        # Release the batch leases now that everything is restored + verified.
        for h in bhandles:
            h.discard()

        # ================================================================== #
        # PHASE E: summary + counter sanity.
        # ================================================================== #
        print("\nPhase E: summary + evict/restore counter sanity")
        sd = off.summary_dict()
        evict_count = sd["evict_count"]
        restore_count = sd["restore_count"]
        d2h = sd["d2h_completed"]
        h2d = sd["h2d_completed"]
        print(f"  evict_count={evict_count} restore_count={restore_count} "
              f"d2h_completed={d2h} h2d_completed={h2d}")

        # Every eviction we issued should have completed its D2H.
        check(evict_count > 0 and d2h == evict_count,
              f"every evict completed its D2H (d2h={d2h}, evict={evict_count})")
        check(restore_count > 0 and h2d == restore_count,
              f"every restore completed its H2D (h2d={h2d}, restore={restore_count})")
        # In this workload we restore essentially everything we evict once, so
        # the counts should be close ("evict_count == restore-ish counts").
        skew = abs(evict_count - restore_count)
        check(skew <= max(2, int(0.2 * evict_count)),
              f"evict_count ~= restore_count (skew {skew} within tolerance)")

        s = off.summary()
        check(isinstance(s, str) and len(s) > 0, "summary() returns non-empty string")
        print("---- summary ----")
        print(s)

    print()
    if failures == 0:
        print("[PASS] llm_workload: all checks OK")
        return 0
    print(f"[FAIL] llm_workload: {failures} checks failed")
    return 1


def main() -> int:
    args = sys.argv[1:]
    if "--spawn" in args:
        proc, sock = _spawn_daemon()
        try:
            return run(sock)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            try:
                if os.path.exists(sock):
                    os.unlink(sock)
            except Exception:
                pass
    else:
        sock = args[0] if args else "/tmp/fastoffload.sock"
        return run(sock)


if __name__ == "__main__":
    sys.exit(main())
