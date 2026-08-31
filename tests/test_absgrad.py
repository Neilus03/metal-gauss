"""absgrad: per-PIXEL gradient magnitude, accumulated in the backward kernel.

The densification signal in train.py accumulates the per-VIEW norm of d_uv, so
a gaussian straddling an edge gets opposing per-pixel pushes that cancel before
anything sees them -- while the gaussian is plainly under-fitting. absgrad sums
the magnitudes instead, inside the rasteriser, so they cannot cancel.

The defining property, and the thing worth testing, is the triangle
inequality: sum|x_i| >= |sum x_i|. If absgrad ever came out below the norm of
d_uv, the reduction would be summing signed values and the whole point would be
lost.
"""
from __future__ import annotations

import pytest
import torch

from metal_gauss import render

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal backend requires MPS"
)


def _scene(n=300, seed=0):
    torch.manual_seed(seed)
    means = (torch.randn(n, 3) * 0.6 + torch.tensor([0.0, 0.0, 4.0])).to("mps")
    quats = torch.randn(n, 4).to("mps")
    scales = (torch.rand(n, 3) * 0.08 + 0.02).to("mps")
    opac = (torch.rand(n) * 0.7 + 0.15).to("mps")
    cols = torch.rand(n, 3).to("mps")
    return means, quats, scales, opac, cols


def _K(W, H):
    K = torch.eye(3)
    f = 0.8 * max(W, H)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    return K.to("mps")


def _run(n=300):
    W, H = 96, 64
    m, q, s, o, c = _scene(n)
    m = m.clone().requires_grad_(True)
    absbuf = torch.zeros(n, device="mps")
    rgb, _, info = render(m, q, s, o, None, _K(W, H), torch.eye(4, device="mps"),
                          W, H, colors=c, backend="metal", absgrad_out=absbuf)
    rgb.square().mean().backward()
    return absbuf, info


def test_absgrad_buffer_is_populated():
    absbuf, _ = _run()
    assert torch.isfinite(absbuf).all()
    assert (absbuf >= 0).all(), "a sum of magnitudes cannot be negative"
    assert absbuf.sum().item() > 0, "backward wrote nothing into the buffer"


def test_absgrad_dominates_the_signed_norm():
    """sum|x_i| >= |sum x_i|, the triangle inequality.

    This is the whole reason absgrad is a different signal from d_uv. If the
    kernel were reducing signed values this assertion would fail on any
    gaussian whose per-pixel gradients oppose each other.
    """
    n = 300
    absbuf, info = _run(n)
    guv = info["uv"].grad
    assert guv is not None, "uv gradient not retained"
    signed_norm = guv.norm(dim=1)
    slack = absbuf - signed_norm
    assert (slack >= -1e-4).all(), \
        f"absgrad below signed norm by {slack.min().item():.3e}"
    # and it must be STRICTLY larger somewhere, or it is measuring the same thing
    assert (slack > 1e-6).any(), "absgrad never exceeds the signed norm"


def test_absgrad_is_not_requested_by_default():
    """No buffer, no cost beyond the one extra atomic: nothing to accumulate."""
    W, H = 96, 64
    m, q, s, o, c = _scene()
    rgb, _, _ = render(m, q, s, o, None, _K(W, H), torch.eye(4, device="mps"),
                       W, H, colors=c, backend="metal")
    assert torch.isfinite(rgb).all()
