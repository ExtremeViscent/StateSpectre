"""Aliased-flat-buffer offload cycle test (needs a GPU + a running daemon).

Reproduces the Megatron/FSDP pattern the integration team flagged: a flat buffer
whose storage is aliased by many views (like buffer.param_data with per-param
views). Destructive evict is wrong there; the correct cycle is non-destructive
off.copy() + storage().resize_(0), then storage().resize_(nbytes) +
handle.restore_into(). This test asserts VRAM is actually freed, aliases stay
valid, and bytes round-trip byte-exact.

Usage: python test_aliased_restore_into.py <unix_sock>
"""

import sys
import torch
import fastoffload as fo


def main():
    sock = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fastoffload.sock"
    dev = "cuda:0"

    with fo.offload_context(daemon_addr=f"unix://{sock}", device=dev, rank=0) as off:
        # A flat buffer with two aliasing views into its storage (the shape of
        # Megatron DDP buffer.param_data + its per-param views).
        flat = torch.arange(4096, device=dev, dtype=torch.float32)  # 16 KiB
        ref = flat.clone().cpu()
        view0 = flat[:2048]          # aliases flat's storage (offset 0)
        view1 = flat[2048:]          # aliases flat's storage (offset 2048)
        storage = flat.untyped_storage()
        nbytes = flat.numel() * flat.element_size()
        assert view0.untyped_storage().data_ptr() == storage.data_ptr()

        # --- offload half: non-destructive copy, then free the storage ---
        h = off.copy(flat, wait=True)                 # D2H; flat untouched
        assert flat.numel() == 4096, "copy must be non-destructive"
        storage.resize_(0)                            # free VRAM, keep Storage object
        assert flat.untyped_storage().size() == 0
        assert view0.untyped_storage().size() == 0, "aliases share the freed storage"
        print(f"[test] offloaded {nbytes} B; storage freed, aliases still bound", flush=True)

        # --- restore half: realloc the SAME storage, H2D back into it ---
        storage.resize_(nbytes)                       # same Storage object, new alloc
        h.restore_into(flat)                          # in-place H2D
        torch.cuda.synchronize()

        # flat is byte-exact ...
        assert torch.equal(flat.cpu(), ref), "flat round-trip mismatch"
        # ... and the aliasing views see the restored data through the shared storage.
        assert view0.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
        assert torch.equal(view0.cpu(), ref[:2048]), "view0 alias broken"
        assert torch.equal(view1.cpu(), ref[2048:]), "view1 alias broken"
        print("[test] restore_into byte-exact; aliases intact", flush=True)

        # restore_into rejects a size mismatch.
        try:
            h.restore_into(torch.empty(10, device=dev, dtype=torch.float32))
            raise AssertionError("expected size-mismatch rejection")
        except ValueError:
            pass

    print("PASS: aliased flat-buffer copy + resize(0) + restore_into")


if __name__ == "__main__":
    main()
