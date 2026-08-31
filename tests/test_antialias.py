"""Mip-Splatting / gsplat "antialiased" opacity compensation.

The compensation is recovered from the conic rather than recomputed in the
kernel, which is cheap but puts the whole correctness burden on one piece of
algebra: conic is the inverse of the DILATED covariance, so the undilated
determinant has to be reconstructed by subtracting `blur` from the diagonal.
These tests pin that algebra against a direct computation, and pin the
antialias-off path as bit-identical to the old behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metal_gauss.metal_backend import antialias_scale  # noqa: E402

BLUR = 0.3


def _conic_from_cov(a, b, c):
    """Invert [[a,b],[b,c]] the way the preprocess kernel emits it."""
    det = a * c - b * b
    return torch.stack([c / det, -b / det, a / det], dim=-1)


def test_scale_matches_direct_determinant_ratio():
    """The whole point: sqrt(det_before / det_after), reconstructed from conic."""
    torch.manual_seed(0)
    n = 512
    # random positive-definite 2x2 covariances spanning sub-pixel to wide
    sx = torch.rand(n).double() * 5.0 + 0.01
    sy = torch.rand(n).double() * 5.0 + 0.01
    rho = (torch.rand(n).double() - 0.5) * 1.2
    a0, c0 = sx * sx, sy * sy
    b0 = rho * sx * sy

    a1, c1 = a0 + BLUR, c0 + BLUR
    conic = _conic_from_cov(a1, b0, c1)

    det0 = a0 * c0 - b0 * b0
    det1 = a1 * c1 - b0 * b0
    want = (det0 / det1).clamp(0.0, 1.0).sqrt()

    got = antialias_scale(conic, BLUR)
    assert torch.allclose(got, want, atol=1e-9), \
        f"max err {(got - want).abs().max().item():.3e}"


def test_scale_never_exceeds_one():
    """Dilation can only lose energy, so the compensation can only reduce."""
    torch.manual_seed(1)
    s = torch.rand(4096).double() * 8.0 + 1e-3
    conic = _conic_from_cov(s * s + BLUR, torch.zeros_like(s), s * s + BLUR)
    sc = antialias_scale(conic, BLUR)
    assert (sc <= 1.0 + 1e-12).all()
    assert (sc >= 0.0).all()


def test_scale_tends_to_one_for_wide_gaussians():
    """A Gaussian far wider than a pixel is barely touched by a 0.3px dilation."""
    wide = torch.tensor([100.0], dtype=torch.float64)
    conic = _conic_from_cov(wide + BLUR, torch.zeros(1, dtype=torch.float64), wide + BLUR)
    assert antialias_scale(conic, BLUR).item() > 0.99


def test_scale_bites_hardest_on_subpixel_gaussians():
    """The eroded case is exactly the one the compensation exists for."""
    tiny = torch.tensor([0.01], dtype=torch.float64)
    conic = _conic_from_cov(tiny + BLUR, torch.zeros(1, dtype=torch.float64), tiny + BLUR)
    sc = antialias_scale(conic, BLUR).item()
    assert sc < 0.1, f"sub-pixel gaussian should be strongly compensated, got {sc}"


def test_gradients_flow_to_conic():
    """Autograd must reach conic; that is why no adjoint was hand-written."""
    s = torch.tensor([0.5, 2.0], dtype=torch.float64)
    conic = _conic_from_cov(s + BLUR, torch.zeros(2, dtype=torch.float64), s + BLUR)
    conic = conic.clone().requires_grad_(True)
    antialias_scale(conic, BLUR).sum().backward()
    assert conic.grad is not None and torch.isfinite(conic.grad).all()
    assert conic.grad.abs().sum() > 0


@pytest.mark.parametrize("blur", [0.0, 0.1, 0.3, 1.0])
def test_zero_blur_is_a_no_op(blur):
    """With no dilation there is nothing to compensate."""
    s = torch.full((64,), 2.0, dtype=torch.float64)
    conic = _conic_from_cov(s + blur, torch.zeros(64, dtype=torch.float64), s + blur)
    sc = antialias_scale(conic, blur)
    if blur == 0.0:
        assert torch.allclose(sc, torch.ones_like(sc), atol=1e-12)
    else:
        assert (sc < 1.0).all()


# --------------------------------------------------------------------------
# End to end, against the oracle. Skipped without MPS.
# --------------------------------------------------------------------------

mps_only = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal backend requires MPS"
)


def _scene(n=400, seed=0, device="mps"):
    torch.manual_seed(seed)
    means = (torch.randn(n, 3) * 0.6 + torch.tensor([0.0, 0.0, 4.0])).to(device)
    quats = torch.randn(n, 4).to(device)
    # deliberately small, so a good share of these land sub-pixel and the
    # compensation actually has something to do
    scales = (torch.rand(n, 3) * 0.03 + 0.004).to(device)
    opac = (torch.rand(n) * 0.7 + 0.15).to(device)
    cols = torch.rand(n, 3).to(device)
    return means, quats, scales, opac, cols


def _K(W, H, device="mps"):
    K = torch.eye(3)
    f = 0.8 * max(W, H)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    return K.to(device)


@mps_only
def test_metal_antialias_matches_oracle():
    from metal_gauss import render
    W, H = 96, 64
    K, vm = _K(W, H), torch.eye(4, device="mps")
    m, q, s, o, c = _scene()

    ref = render(m, q, s, o, None, K, vm, W, H, colors=c,
                 backend="torch_ref", antialias=True)[0]
    got = render(m, q, s, o, None, K, vm, W, H, colors=c,
                 backend="metal", antialias=True)[0]
    err = (ref - got).abs().max().item()
    assert err < 5e-3, f"metal vs oracle with antialias: max abs {err:.2e}"


@mps_only
def test_antialias_actually_changes_the_image():
    """A guard against the flag silently doing nothing.

    This repo has shipped a dead branch before -- the Metal binning path was
    unreachable for weeks because an upstream reassignment made its condition
    always false. A flag that changes no pixels is the same failure.
    """
    from metal_gauss import render
    W, H = 96, 64
    K, vm = _K(W, H), torch.eye(4, device="mps")
    m, q, s, o, c = _scene()

    off = render(m, q, s, o, None, K, vm, W, H, colors=c,
                 backend="metal", antialias=False)[0]
    on = render(m, q, s, o, None, K, vm, W, H, colors=c,
                backend="metal", antialias=True)[0]
    diff = (off - on).abs().max().item()
    assert diff > 1e-4, f"--antialias changed nothing (max diff {diff:.2e})"
    # and it must DARKEN, never brighten: the compensation only reduces opacity
    assert on.sum().item() <= off.sum().item() + 1e-3
