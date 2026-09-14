"""Graft a trained stop head onto a holds checkpoint that never had one.

Tests whether the head learned something GENERAL about the action-token space
or something specific to the run it trained on. The head is 20K parameters
reading pooled action tokens, so architecturally it transfers between any two
checkpoints; whether it transfers FUNCTIONALLY depends on whether the two runs'
feature spaces stayed aligned after diverging from their shared base.

Same head type (MLP -> MLP) is the plausible case. Across head types
(MLP -> smolVLA) the action tokens are produced by different modules entirely,
so a transfer there is expected to fail -- worth running only as a control.
"""
from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="checkpoint WITH a trained stop head")
    ap.add_argument("--recipient", required=True, help="holds checkpoint without one")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    don = torch.load(args.donor, map_location="cpu", weights_only=False, mmap=True)
    rec = torch.load(args.recipient, map_location="cpu", weights_only=False, mmap=True)
    dsd, rsd = don["state_dict"], rec["state_dict"]

    keys = [k for k in dsd if "stop_head" in k]
    if not keys:
        raise SystemExit("donor has no stop_head weights")
    print(f"  donor stop_head tensors: {len(keys)}")
    for k in keys:
        print(f"    {k:52s} {tuple(dsd[k].shape)}  absmean={dsd[k].float().abs().mean():.5f}")

    clash = [k for k in keys if k in rsd]
    if clash:
        print(f"  recipient ALREADY has {len(clash)} of these; they will be overwritten")

    new = dict(rsd)
    for k in keys:
        new[k] = dsd[k].clone()
    rec["state_dict"] = new
    # the recipient's optimizer state has no slots for these params; drop it so
    # the loader warm-starts instead of crashing on a param-group mismatch.
    rec["optimizer_states"] = []
    torch.save(rec, args.out)
    print(f"\n  wrote {args.out}")
    print(f"  recipient keys: {len(rsd)} -> {len(new)}")


if __name__ == "__main__":
    main()
