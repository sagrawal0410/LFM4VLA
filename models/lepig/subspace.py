"""Local SGD-trajectory PCA basis (SWAG-style) for the posterior subspace.

Keeps S snapshots of the *selected* parameters over a trailing window, centres
them, and takes the top-r principal directions of the deltas. The basis is
frozen for the lifetime of one scoring snapshot and rebuilt only on refresh.

Scalability rules from the plan, which matter for the A3 (full-network)
variant: never gather the whole model onto one rank. Deltas stay in the same
shard layout as training, the S x S Gram matrix is formed with distributed dot
products, only the small matrix is eigendecomposed, and basis vectors are
reconstructed shard-wise when JVPs are evaluated. Deltas are stored BF16 on
CPU; all curvature algebra runs FP32.
"""
from __future__ import annotations

from typing import Iterable, List

import torch


class TrajectorySubspace:
    def __init__(self, rank: int = 12, n_snapshots: int = 13,
                 window_steps: int = 2000, store_dtype=torch.bfloat16):
        self.rank = int(rank)
        self.n_snapshots = int(n_snapshots)
        self.window_steps = int(window_steps)
        self.every = max(1, self.window_steps // max(self.n_snapshots - 1, 1))
        self.store_dtype = store_dtype
        self._snaps: List[List[torch.Tensor]] = []      # CPU, bf16, sharded
        self._coeffs: torch.Tensor | None = None        # [S, r] eigenvectors
        self._mean: List[torch.Tensor] | None = None

    # -- snapshot capture --------------------------------------------------
    def maybe_capture(self, step: int, params: Iterable[torch.nn.Parameter]):
        if step % self.every:
            return False
        self.capture(params)
        return True

    def capture(self, params: Iterable[torch.nn.Parameter]):
        snap = [p.detach().to("cpu", self.store_dtype, copy=True)
                for p in params]
        self._snaps.append(snap)
        if len(self._snaps) > self.n_snapshots:
            self._snaps.pop(0)
        self._coeffs = None          # invalidate the basis

    @property
    def ready(self) -> bool:
        return len(self._snaps) >= max(3, self.rank // 4)

    # -- basis construction -------------------------------------------------
    def build(self, all_reduce=None):
        """Eigendecompose the S x S Gram matrix of centred deltas.

        all_reduce: optional callable summing a scalar tensor across ranks, so
        the Gram entries are global even though each rank holds only a shard.
        """
        if not self.ready:
            return False
        S = len(self._snaps)
        self._mean = [torch.stack([s[i].float() for s in self._snaps]).mean(0)
                      for i in range(len(self._snaps[0]))]
        gram = torch.zeros(S, S, dtype=torch.float32)
        for a in range(S):
            for b in range(a, S):
                dot = torch.zeros((), dtype=torch.float32)
                for i in range(len(self._mean)):
                    da = self._snaps[a][i].float() - self._mean[i]
                    db = self._snaps[b][i].float() - self._mean[i]
                    dot += (da * db).sum()
                if all_reduce is not None:
                    dot = all_reduce(dot)
                gram[a, b] = gram[b, a] = dot
        evals, evecs = torch.linalg.eigh(gram)           # ascending
        idx = torch.argsort(evals, descending=True)[: self.rank]
        sel = evecs[:, idx]                              # [S, r]
        # normalise so each basis direction has unit norm in parameter space
        scale = evals[idx].clamp_min(1e-12).sqrt()
        self._coeffs = (sel / scale.unsqueeze(0)).to(torch.float32)
        self._eigvals = evals[idx].clone()
        return True

    @property
    def effective_rank(self) -> int:
        return 0 if self._coeffs is None else int(self._coeffs.shape[1])

    def direction(self, k: int, like: Iterable[torch.nn.Parameter]) -> List[torch.Tensor]:
        """Reconstruct basis vector k shard-wise, matching `like`'s layout."""
        if self._coeffs is None:
            raise RuntimeError("build() must run before direction()")
        c = self._coeffs[:, k]
        out = []
        for i, p in enumerate(like):
            acc = torch.zeros_like(self._mean[i])
            for s in range(len(self._snaps)):
                acc += c[s] * (self._snaps[s][i].float() - self._mean[i])
            out.append(acc.to(device=p.device, dtype=torch.float32))
        return out
