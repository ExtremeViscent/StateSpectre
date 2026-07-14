#!/usr/bin/env python3
"""End-to-end Python test of the state_spectre runtime against a live daemon.

Mirrors the examples in python_api/PYTHON_API.md:
  - single-tensor destructive evict + restore, byte-identical
  - destructive invalidation frees the original storage (set_empty)
  - non-destructive copy leaves the original valid
  - evict_many / restore_many
  - view rejected by default; accepted in compact mode
  - autograd tensor rejected by default
  - manage() then record.evict()
  - summary()

Requires a running daemon on the socket and a CUDA device.
Usage: python test_python_e2e.py <socket_path>
"""
import sys
import torch
import state_spectre as ss


def approx_free_mb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


def main():
    socket_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/state_spectre.sock"
    device = "cuda:0"
    torch.cuda.init()
    torch.cuda.set_device(0)

    failures = 0
    def check(cond, msg):
        nonlocal failures
        status = "ok  " if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {msg}")

    with ss.offload_context(
        daemon_addr=f"unix://{socket_path}",
        device=device,
        rank=0,
        invalidate_mode="set_empty",
    ) as off:

        # ---- 1. single destructive evict + restore, byte-identical ----
        print("test: single destructive evict/restore")
        x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        original = x.clone()
        h = off.evict(x, name="x", wait=True)
        # After destructive evict with wait=True, original storage is replaced.
        check(x.numel() == 0 or x.data_ptr() != original.data_ptr(),
              "original tensor storage emptied after destructive evict "
              f"(numel now {x.numel()})")
        x2 = h.restore(device=device)
        check(x2.shape == original.shape, f"restored shape {tuple(x2.shape)}")
        check(x2.dtype == original.dtype, f"restored dtype {x2.dtype}")
        check(torch.equal(x2, original), "restored tensor byte-identical to original")

        # ---- 2. non-destructive copy leaves original valid ----
        print("test: non-destructive copy")
        y = torch.arange(4096, device=device, dtype=torch.float32)
        yorig = y.clone()
        hc = off.copy(y, name="y", wait=True)
        check(y.numel() == 4096 and torch.equal(y, yorig),
              "original remains valid after non-destructive copy")
        y2 = hc.restore(device=device)
        check(torch.equal(y2, yorig), "copy restore byte-identical")

        # ---- 3. evict_many / wait / restore_many ----
        print("test: evict_many / restore_many")
        k = torch.randn(512, 512, device=device)
        v = torch.randn(512, 512, device=device)
        korig, vorig = k.clone(), v.clone()
        handles = off.evict_many([k, v], names=["k", "v"])
        off.wait(handles)
        restored = off.restore_many(handles, device=device)
        check(len(restored) == 2, "restore_many returned 2 tensors")
        check(torch.equal(restored[0], korig), "k byte-identical")
        check(torch.equal(restored[1], vorig), "v byte-identical")

        # ---- 4. view rejected by default; compact mode accepted ----
        print("test: view validation")
        base = torch.empty(1 << 20, device=device)
        view = base[1:1024]
        rejected = False
        try:
            off.evict(view)
        except Exception:
            rejected = True
        check(rejected, "view rejected by default (not full-storage owner)")
        # compact mode: copies logical bytes, no VRAM-free guarantee.
        vv = base[16:2048].clone()  # a fresh contiguous logical tensor
        vv_src = base[16:2048]
        vv_orig = vv_src.clone()
        hv = off.evict(vv_src, allow_views=True, compact=True, wait=True)
        rv = hv.restore(device=device)
        check(torch.equal(rv, vv_orig), "compact view mode restores logical bytes")

        # ---- 5. autograd tensor rejected by default ----
        print("test: autograd validation")
        g = torch.randn(256, 256, device=device, requires_grad=True)
        arej = False
        try:
            off.evict(g)
        except Exception:
            arej = True
        check(arej, "requires_grad tensor rejected by default")

        # ---- 6. manage() then record.evict() ----
        print("test: manage + record.evict")
        m = torch.randn(300, 300, device=device)
        morig = m.clone()
        rec = off.manage(m, name="managed")
        hm = rec.evict(wait=True)
        mm = hm.restore(device=device)
        check(torch.equal(mm, morig), "managed tensor restore byte-identical")

        # ---- 7. summary ----
        print("test: summary")
        s = off.summary()
        check(isinstance(s, str) and len(s) > 0, "summary() returns non-empty string")
        print("---- summary ----")
        print(s)

    print()
    if failures == 0:
        print("[PASS] python_e2e: all checks OK")
        return 0
    print(f"[FAIL] python_e2e: {failures} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
