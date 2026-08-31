"""Trainer-component tests: SSIM vs skimage, MCMC invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@mps
def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity

    from metal_gauss.train import _gaussian_kernel, ssim

    rng = np.random.default_rng(0)
    a = rng.random((64, 96, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.1, a.shape).astype(np.float32), 0, 1)

    ours = ssim(torch.tensor(a, device="mps"), torch.tensor(b, device="mps"),
                _gaussian_kernel(device="mps")).item()
    theirs = structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                   gaussian_weights=True, sigma=1.5,
                                   use_sample_covariance=False)
    assert abs(ours - theirs) < 0.02, f"{ours} vs skimage {theirs}"


def _params(n=500, dead_frac=0.3, device="cpu"):
    g = torch.Generator().manual_seed(0)
    logit = torch.full((n,), 2.0)
    dead = torch.randperm(n, generator=g)[: int(n * dead_frac)]
    logit[dead] = -8.0                       # sigmoid ~ 0.0003 < thresh
    return {
        "means": torch.randn(n, 3, generator=g),
        "log_scales": torch.randn(n, 3, generator=g) * 0.1 - 3,
        "quats": torch.randn(n, 4, generator=g),
        "logit_opac": logit,
        "sh": torch.randn(n, 16, 3, generator=g),
    }, dead


def test_relocate_preserves_budget_and_revives_dead():
    from metal_gauss.mcmc import relocate

    p, dead = _params()
    n = len(p["means"])
    moved = relocate(p)
    assert moved == len(dead)
    for k, t in p.items():
        assert len(t) == n, f"{k} changed size"     # fixed budget: never grows
    opac = torch.sigmoid(p["logit_opac"])
    assert (opac[dead] > 0.005).all(), "relocated gaussians must be alive"


def test_relocate_noop_when_all_alive():
    from metal_gauss.mcmc import relocate

    p, _ = _params(dead_frac=0.0)
    before = {k: t.clone() for k, t in p.items()}
    assert relocate(p) == 0
    for k in p:
        assert torch.equal(p[k], before[k])


def test_noise_only_moves_transparent_gaussians():
    from metal_gauss.mcmc import add_noise

    p, dead = _params()
    before = p["means"].clone()
    add_noise(p, lr_means=2e-4)
    delta = (p["means"] - before).norm(dim=1)
    alive = torch.ones(len(delta), dtype=bool)
    alive[dead] = False
    # transparent gaussians random-walk, opaque ones barely move
    assert delta[dead].mean() > 50 * max(delta[alive].mean().item(), 1e-12)


def test_selective_adam_matches_dense_on_fully_visible():
    """With everything visible, SelectiveAdam must equal torch.optim.Adam."""
    from metal_gauss.selective_adam import SelectiveAdam

    torch.manual_seed(0)
    a = torch.randn(50, 3, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    dense = torch.optim.Adam([b], lr=1e-2, eps=1e-15)
    sel = SelectiveAdam([{"params": [a], "lr": 1e-2}], eps=1e-15)
    vis = torch.ones(50, dtype=torch.bool)

    for _ in range(5):
        for t in (a, b):
            t.grad = None
        (a.square().sum()).backward()
        (b.square().sum()).backward()
        sel.step(vis)
        dense.step()
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_selective_adam_leaves_invisible_untouched():
    from metal_gauss.selective_adam import SelectiveAdam

    torch.manual_seed(1)
    t = torch.randn(40, 3, requires_grad=True)
    before = t.detach().clone()
    sel = SelectiveAdam([{"params": [t], "lr": 1e-2}])
    vis = torch.zeros(40, dtype=torch.bool)
    vis[:10] = True
    t.grad = torch.randn(40, 3)
    sel.step(vis)
    assert not torch.allclose(t[:10], before[:10])
    assert torch.equal(t[10:], before[10:])


# ---------------------------------------------------------------- MCMC math

def test_relocation_opacity_composites_back():
    """N copies at o_new must composite back to the original opacity."""
    from metal_gauss.mcmc import relocate

    torch.manual_seed(0)
    n = 200
    p = {
        "means": torch.randn(n, 3), "quats": torch.randn(n, 4),
        "log_scales": torch.full((n, 3), -3.0), "sh": torch.randn(n, 16, 3),
        "logit_opac": torch.full((n,), 2.0),
    }
    p["logit_opac"][:120] = -8.0            # 120 dead onto 80 live -> N>2 forced
    o_before = torch.sigmoid(p["logit_opac"][120:]).clone()
    relocate(p)
    o_after = torch.sigmoid(p["logit_opac"])
    # every gaussian is now alive
    assert (o_after > 0.005).all()
    # and the *stack* is not systematically more opaque than the original
    assert o_after.max() <= o_before.max() + 1e-5


def test_relocation_handles_duplicate_picks():
    """Regression: multinomial(replacement=True) repeats must not last-write-win.

    With many dead and one live gaussian, every dead index picks the same
    target. The correct result is an N-way split (all copies much fainter
    than the source); the buggy scatter produced near-full-opacity clones.
    """
    from metal_gauss.mcmc import relocate

    n = 33
    p = {
        "means": torch.zeros(n, 3), "quats": torch.zeros(n, 4),
        "log_scales": torch.full((n, 3), -3.0), "sh": torch.zeros(n, 16, 3),
        "logit_opac": torch.full((n,), -8.0),
    }
    p["logit_opac"][0] = 4.0                      # exactly one live target
    o_src = torch.sigmoid(p["logit_opac"][0]).item()
    relocate(p)
    o = torch.sigmoid(p["logit_opac"])
    assert (o > 0.005).all(), "all should be revived"
    # N=33 way split: each copy must be far fainter than the source
    assert o.max() < o_src * 0.5, f"max {o.max():.3f} vs source {o_src:.3f}"


def test_relocation_zeroes_adam_state():
    from metal_gauss.mcmc import relocate

    n = 60
    p = {
        "means": torch.randn(n, 3, requires_grad=True),
        "quats": torch.randn(n, 4, requires_grad=True),
        "log_scales": torch.full((n, 3), -3.0, requires_grad=True),
        "sh": torch.randn(n, 16, 3, requires_grad=True),
        "logit_opac": torch.full((n,), 2.0, requires_grad=True),
    }
    with torch.no_grad():
        p["logit_opac"][:20] = -8.0
    opt = torch.optim.Adam(list(p.values()), lr=1e-3)
    for t in p.values():
        t.grad = torch.randn_like(t)
    opt.step()
    assert opt.state[p["means"]]["exp_avg"].abs().sum() > 0
    with torch.no_grad():
        relocate(p, opt=opt)
    dead_rows = opt.state[p["means"]]["exp_avg"][:20]
    assert dead_rows.abs().sum() == 0, "relocated rows must have zeroed momentum"


def test_grow_ramps_active_count():
    from metal_gauss.mcmc import grow

    n = 1000
    p = {
        "means": torch.randn(n, 3), "quats": torch.randn(n, 4),
        "log_scales": torch.full((n, 3), -3.0), "sh": torch.randn(n, 16, 3),
        "logit_opac": torch.full((n,), 2.0),
    }
    new_active = grow(p, target=400, active=200)
    assert new_active == 400
    assert (torch.sigmoid(p["logit_opac"][200:400]) > 0.005).all()
    # capped at the preallocated size
    assert grow(p, target=5000, active=400) == n


def test_noise_anneals_with_lr():
    from metal_gauss.mcmc import add_noise

    n = 500
    def fresh():
        torch.manual_seed(3)
        return {
            "means": torch.zeros(n, 3),
            "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
            "log_scales": torch.full((n, 3), -3.0),
            "sh": torch.zeros(n, 16, 3),
            "logit_opac": torch.full((n,), -8.0),      # all transparent
        }
    hot, cold = fresh(), fresh()
    add_noise(hot, lr_means=2e-4)
    add_noise(cold, lr_means=2e-6)                     # 100x decayed
    assert hot["means"].norm() > 50 * cold["means"].norm()
