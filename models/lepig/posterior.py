"""Subspace Laplace posterior and predictive information gain.

All plans share this machinery; only the predictive functional g_i changes.
Everything is kept in the r x r subspace (r = 12 by default), so no P x P
covariance is ever formed and no full Jacobian is ever materialised.

Notation follows the plan document:
    phi = phi_hat + B alpha,  alpha in R^r
    g_i(alpha) ~= mu_i + G_i alpha,      G_i = d g_i / d alpha   [d_g, r]
    F_i        = G_i^T R_i^-1 G_i                                [r, r]
    Sigma_D    = (Lambda_0 + sum_j F_j)^-1
    S_e(D)     = R_e + G_e Sigma_D G_e^T
    PIG(i)     = 1/M sum_e 1/2 [ logdet S_e(D) - logdet S_e(D+i) ]

`B B^T` is NOT a full posterior covariance: epistemic uncertainty is
represented only inside the chosen subspace. Nothing here claims otherwise.
"""
from __future__ import annotations

import torch


def _chol_logdet(mat: torch.Tensor, jitter_rel: float = 1e-5) -> torch.Tensor:
    """logdet of a symmetric PD matrix via Cholesky, with relative jitter."""
    m = 0.5 * (mat + mat.transpose(-1, -2))
    scale = torch.diagonal(m, dim1=-2, dim2=-1).abs().mean().clamp_min(1e-12)
    eye = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
    for k in range(6):                      # escalate jitter only if needed
        try:
            L = torch.linalg.cholesky(m + (jitter_rel * (10 ** k)) * scale * eye)
            return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        except RuntimeError:
            continue
    return torch.linalg.slogdet(m)[1]


class SubspacePosterior:
    """Gaussian posterior over alpha in R^r, held as a precision matrix."""

    def __init__(self, rank: int, prior_precision: float = 1.0,
                 jitter_rel: float = 1e-5, device=None, dtype=torch.float32):
        self.r = int(rank)
        self.jitter_rel = float(jitter_rel)
        self.dtype = dtype
        eye = torch.eye(self.r, device=device, dtype=dtype)
        self.prior_precision = float(prior_precision)
        self.Lambda = self.prior_precision * eye          # Lambda_0

    # -- curvature accumulation ------------------------------------------
    def reset(self):
        eye = torch.eye(self.r, device=self.Lambda.device, dtype=self.dtype)
        self.Lambda = self.prior_precision * eye

    def add_fisher(self, F: torch.Tensor):
        """Accumulate one calibration example's Fisher block [r, r]."""
        self.Lambda = self.Lambda + F.to(self.Lambda)

    @staticmethod
    def fisher_from_jac(G: torch.Tensor, obs_var: float = 1.0) -> torch.Tensor:
        """F = G^T R^-1 G with R = obs_var * I.  G: [d_g, r]."""
        G = G.to(torch.float32)
        return (G.transpose(-1, -2) @ G) / float(obs_var)

    @property
    def Sigma(self) -> torch.Tensor:
        eye = torch.eye(self.r, device=self.Lambda.device, dtype=self.dtype)
        scale = torch.diagonal(self.Lambda).abs().mean().clamp_min(1e-12)
        return torch.linalg.solve(self.Lambda + self.jitter_rel * scale * eye, eye)

    # -- predictive information gain --------------------------------------
    def pig(self, G_cand: torch.Tensor, G_anchors: list[torch.Tensor],
            obs_var: float = 1.0) -> torch.Tensor:
        """PIG of one candidate against the anchor bank.

        G_cand:    [d_g, r]        candidate Jacobian
        G_anchors: list of [d_e, r] anchor Jacobians (cached per snapshot)
        """
        Sig = self.Sigma
        F_i = self.fisher_from_jac(G_cand, obs_var)
        # Sigma_{D+i} = (Lambda + F_i)^-1, still r x r
        eye = torch.eye(self.r, device=Sig.device, dtype=Sig.dtype)
        scale = torch.diagonal(self.Lambda).abs().mean().clamp_min(1e-12)
        Sig_post = torch.linalg.solve(
            self.Lambda + F_i + self.jitter_rel * scale * eye, eye)
        total = Sig.new_zeros(())
        for G_e in G_anchors:
            Ge = G_e.to(Sig)
            R = obs_var * torch.eye(Ge.shape[0], device=Sig.device, dtype=Sig.dtype)
            S_pre = R + Ge @ Sig @ Ge.transpose(-1, -2)
            S_post = R + Ge @ Sig_post @ Ge.transpose(-1, -2)
            total = total + 0.5 * (_chol_logdet(S_pre, self.jitter_rel)
                                   - _chol_logdet(S_post, self.jitter_rel))
        return total / max(len(G_anchors), 1)

    # -- diagnostics used by the falsification checklist -------------------
    def raw_epistemic(self, G: torch.Tensor) -> torch.Tensor:
        """tr(G Sigma G^T) -- the 'RawEpi' baseline score."""
        Gm = G.to(self.Sigma)
        return torch.einsum("ij,jk,ik->", Gm, self.Sigma, Gm)

    def parameter_ig(self, G: torch.Tensor, obs_var: float = 1.0) -> torch.Tensor:
        """0.5 logdet(I + Sigma F_i) -- the 'ParameterIG' baseline score."""
        F = self.fisher_from_jac(G, obs_var)
        eye = torch.eye(self.r, device=F.device, dtype=F.dtype)
        return 0.5 * _chol_logdet(eye + self.Sigma @ F, self.jitter_rel)
