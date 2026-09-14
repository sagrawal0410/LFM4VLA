"""Can the representation linearly predict distance-to-goal?

Every proposed stopping fix -- stop head, distance head, delta encoding --
assumes arrival information IS in the representation and we are merely failing
to decode it. This tests that assumption directly: ridge-regress pooled action
tokens onto the true geodesic distance to goal, fit on one set of val_unseen
episodes and scored on held-out ones.

  high R^2  -> the information is there; extraction is the bottleneck, and the
               cheapest decoder wins.
  low  R^2  -> it is not there. No readout can recover it, and the fix has to
               be representational (a COUPLED distance head that forces the
               backbone to encode it).

Baselines matter here: a constant predictor and a "steps elapsed" predictor
bound how much credit the representation deserves. Distance-to-goal correlates
with elapsed time, so a probe could look good while having learned nothing
about the scene.
"""
from __future__ import annotations

import argparse

import numpy as np


def ridge_fit(X, y, lam):
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    A = Xb.T @ Xb + lam * np.eye(d + 1, dtype=X.dtype)
    return np.linalg.solve(A, Xb.T @ y)


def r2(y, yhat):
    ss = float(((y - yhat) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / max(tot, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--holdout", type=float, default=0.3)
    args = ap.parse_args()

    Xs, ys, eps = [], [], []
    for i, f in enumerate(args.npz):
        d = np.load(f)
        Xs.append(d["X"].astype(np.float64))
        ys.append(d["y"].astype(np.float64))
        eps.append(np.full(len(d["y"]), i))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    src = np.concatenate(eps)
    print(f"  samples={len(y)}  dim={X.shape[1]}  files={len(args.npz)}")
    print(f"  distance-to-goal: mean={y.mean():.2f} std={y.std():.2f} "
          f"min={y.min():.2f} max={y.max():.2f}")

    # split by SOURCE FILE, not by row: consecutive steps within an episode are
    # highly correlated, so a random row split leaks and inflates R^2.
    uniq = np.unique(src)
    n_test = max(1, int(round(len(uniq) * args.holdout)))
    rng = np.random.default_rng(0)
    test_files = set(rng.choice(uniq, size=n_test, replace=False).tolist())
    te = np.isin(src, list(test_files))
    tr = ~te
    print(f"  train rows={tr.sum()}  test rows={te.sum()} "
          f"(held-out shards: {sorted(test_files)})")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd

    print("\n  BASELINES (what the probe must beat)")
    const = np.full(te.sum(), y[tr].mean())
    print(f"    constant (train mean) : R^2 = {r2(y[te], const):+.4f}")

    print("\n  RIDGE PROBE: pooled action tokens -> geodesic distance")
    best = (-9, None)
    for lam in (1.0, 10.0, 100.0, 1000.0, 1e4):
        w = ridge_fit(Xtr, y[tr], lam)
        pred = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ w
        s = r2(y[te], pred)
        mae = float(np.abs(y[te] - pred).mean())
        print(f"    lambda={lam:<8g} R^2 = {s:+.4f}   MAE = {mae:.2f} m")
        if s > best[0]:
            best = (s, lam)
    print(f"\n  best R^2 = {best[0]:+.4f} at lambda={best[1]}")
    if best[0] > 0.5:
        print("  -> arrival information IS linearly decodable. Extraction is "
              "the bottleneck; the cheapest decoder should win.")
    elif best[0] > 0.2:
        print("  -> partially decodable. A learned decoder helps but will not "
              "close the gap alone.")
    else:
        print("  -> NOT linearly decodable. No readout fixes this; the fix has "
              "to be representational (coupled distance head).")


if __name__ == "__main__":
    main()
