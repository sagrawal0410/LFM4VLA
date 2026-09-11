"""Falsification checklist for the LEPIG machinery (plan doc section 9).

These must pass before any full training run. They are pure-CPU and take
seconds; none of them needs the VLA.
"""
from __future__ import annotations

import math
import torch

from models.lepig.posterior import SubspacePosterior, _chol_logdet
from models.lepig.routing import robust_weight, grad_scale_identity

torch.manual_seed(0)
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


# 1 ---- exact linear-Gaussian PIG: closed form vs our implementation -------
def test_exact_linear_gaussian():
    r, d = 5, 3
    post = SubspacePosterior(rank=r, prior_precision=1.0, jitter_rel=0.0)
    Gi = torch.randn(d, r); Ge = torch.randn(d, r)
    Sig = post.Sigma
    Fi = Gi.T @ Gi
    Sig_post = torch.linalg.inv(post.Lambda + Fi)
    I = torch.eye(d)
    ref = 0.5 * (torch.logdet(I + Ge @ Sig @ Ge.T)
                 - torch.logdet(I + Ge @ Sig_post @ Ge.T))
    got = post.pig(Gi, [Ge])
    check("exact linear-Gaussian PIG matches closed form",
          torch.allclose(ref, got, atol=1e-5), f"ref={ref:.6f} got={got:.6f}")


# 2 ---- JVP vs explicit Jacobian on a tiny network -------------------------
def test_jvp_matches_jacobian():
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(),
                              torch.nn.Linear(6, 3))
    x = torch.randn(1, 4)
    params = [p for p in net.parameters()]
    flat = torch.cat([p.reshape(-1) for p in params])
    B = torch.randn(flat.numel(), 3) * 0.01          # r = 3 subspace

    def f(alpha):
        off, out = 0, []
        vec = flat + B @ alpha
        for p in params:
            n = p.numel(); out.append(vec[off:off + n].view_as(p)); off += n
        return torch.func.functional_call(
            net, {k: v for (k, _), v in zip(net.named_parameters(), out)}, (x,)).reshape(-1)

    a0 = torch.zeros(3)
    J_explicit = torch.autograd.functional.jacobian(f, a0)       # [3, 3]
    cols = []
    for k in range(3):
        e = torch.zeros(3); e[k] = 1.0
        _, jv = torch.func.jvp(f, (a0,), (e,))
        cols.append(jv)
    J_jvp = torch.stack(cols, dim=1)
    check("JVP columns match explicit Jacobian",
          torch.allclose(J_explicit, J_jvp, atol=1e-5),
          f"max|d|={(J_explicit - J_jvp).abs().max():.2e}")


# 3 ---- Cholesky/determinant-lemma equivalence ----------------------------
def test_chol_logdet():
    A = torch.randn(7, 7); M = A @ A.T + 7 * torch.eye(7)
    check("Cholesky logdet == slogdet",
          torch.allclose(_chol_logdet(M, 0.0), torch.linalg.slogdet(M)[1], atol=1e-4))


# 4 ---- orthogonal candidate/anchor gives ~zero predictive IG -------------
def test_orthogonality_zero_ig():
    r = 6
    post = SubspacePosterior(rank=r, prior_precision=1.0, jitter_rel=0.0)
    Gi = torch.zeros(2, r); Gi[0, 0] = 1.0; Gi[1, 1] = 1.0    # spans dims 0,1
    Ge = torch.zeros(2, r); Ge[0, 3] = 1.0; Ge[1, 4] = 1.0    # spans dims 3,4
    pig = post.pig(Gi, [Ge])
    check("orthogonal candidate/anchor -> ~0 predictive IG",
          abs(float(pig)) < 1e-6, f"pig={float(pig):.2e}")


# 5 ---- parameter IG high while predictive IG low for irrelevant anchor ---
def test_param_ig_vs_predictive_ig():
    r = 6
    post = SubspacePosterior(rank=r, prior_precision=1.0, jitter_rel=0.0)
    Gi = torch.zeros(2, r); Gi[0, 0] = 3.0; Gi[1, 1] = 3.0
    Ge = torch.zeros(2, r); Ge[0, 3] = 1.0; Ge[1, 4] = 1.0
    pid = float(post.parameter_ig(Gi)); pig = float(post.pig(Gi, [Ge]))
    check("parameter IG >> predictive IG for irrelevant anchor",
          pid > 1.0 and abs(pig) < 1e-6, f"paramIG={pid:.3f} PIG={pig:.2e}")


# 6 ---- score finite, non-degenerate, monotone in alignment ---------------
def test_score_sane():
    r = 8
    post = SubspacePosterior(rank=r, prior_precision=1.0)
    Ge = torch.randn(4, r)
    aligned = post.pig(Ge.clone(), [Ge])                  # perfectly aligned
    rnd = post.pig(torch.randn(4, r) * 0.01, [Ge])        # tiny, misaligned
    check("PIG finite and positive", torch.isfinite(aligned) and aligned > 0,
          f"aligned={float(aligned):.4f}")
    check("PIG larger for aligned candidate than weak one",
          float(aligned) > float(rnd), f"{float(aligned):.4f} > {float(rnd):.4f}")


# 7 ---- score -> weight transform bounds ----------------------------------
def test_weight_transform():
    s = torch.randn(256) * 3.0
    w = robust_weight(s)
    # The plan clamps to [0.5, 2.0] and THEN mean-normalises, so the final
    # values can sit just outside that box. The invariant preserved by the
    # rescale is the bounded ratio (2.0 / 0.5 = 4) and a bounded absolute span.
    ratio = float(w.max() / w.min())
    check("weight max/min ratio <= clamp ratio (4)", ratio <= 4.0 + 1e-5,
          f"ratio={ratio:.4f}")
    check("weights bounded well away from 0 and from blowing up",
          bool((w > 0.2).all() and (w < 5.0).all()),
          f"min={float(w.min()):.3f} max={float(w.max()):.3f}")
    check("weights mean-normalised to 1", abs(float(w.mean()) - 1.0) < 1e-5,
          f"mean={float(w.mean()):.6f}")
    check("weights detached", not w.requires_grad)
    const = robust_weight(torch.full((64,), 2.5))
    check("degenerate (constant) scores -> all-ones weights",
          torch.allclose(const, torch.ones_like(const), atol=1e-4))


# 8 ---- gradient routing: forward identical, grads scaled ------------------
def test_grad_routing():
    torch.manual_seed(2)
    backbone = torch.nn.Linear(5, 7)
    head = torch.nn.Linear(7, 2)
    x = torch.randn(4, 5)
    w = torch.tensor([0.5, 1.0, 1.5, 2.0])

    H = backbone(x)
    out_plain = head(H)
    H2 = grad_scale_identity(backbone(x), w)
    out_routed = head(H2)
    check("routing leaves forward value unchanged",
          torch.allclose(out_plain, out_routed, atol=1e-6))

    def grads(use_w):
        backbone.zero_grad(); head.zero_grad()
        Hh = backbone(x)
        if use_w:
            Hh = grad_scale_identity(Hh, w)
        # per-example sum so the weight shows up per row
        head(Hh).pow(2).sum().backward()
        return ([p.grad.clone() for p in head.parameters()],
                [p.grad.clone() for p in backbone.parameters()])

    hg0, bg0 = grads(False)
    hg1, bg1 = grads(True)
    # head grads change only because H itself is unchanged -> identical
    check("flow-head grads unchanged by routing",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(hg0, hg1)))
    check("backbone grads changed by routing",
          any(not torch.allclose(a, b, atol=1e-6) for a, b in zip(bg0, bg1)))

    # exact per-example check: w=1 everywhere must reproduce the plain grads
    w_ones = torch.ones(4)
    backbone.zero_grad(); head.zero_grad()
    head(grad_scale_identity(backbone(x), w_ones)).pow(2).sum().backward()
    bg_one = [p.grad.clone() for p in backbone.parameters()]
    check("w=1 routing == no routing (backbone)",
          all(torch.allclose(a, b, atol=1e-6) for a, b in zip(bg0, bg_one)))


if __name__ == "__main__":
    print("=== LEPIG falsification checklist ===")
    for fn in (test_exact_linear_gaussian, test_jvp_matches_jacobian, test_chol_logdet,
               test_orthogonality_zero_ig, test_param_ig_vs_predictive_ig,
               test_score_sane, test_weight_transform, test_grad_routing):
        fn()
    print(f"\n  {len(OK)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  FAILED:", ", ".join(FAIL)); raise SystemExit(1)
    print("  ALL CHECKS PASSED")
