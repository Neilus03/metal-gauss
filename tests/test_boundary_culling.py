"""Splats sitting exactly on tile or screen boundaries must not be dropped.

LichtFeld-Studio issue #1755 reports splats landing bit-exactly on a tile edge
being silently culled. Our binning uses inclusive floor on both ends
(`build_tile_lists`) and our screen tests are strict (`u + rx > 0`), which is
the same shape of code -- so this is checked rather than assumed.

The test does not compare against a reference convention (both backends could
share a wrong one). It sweeps a splat across each boundary and asserts the
rendered energy stays CONTINUOUS: a dropped splat shows up as a step change
far larger than its neighbours.
"""

import numpy as np
import pytest
import torch

from metal_gauss import torch_ref
from metal_gauss.metal_backend import render as metal_render

DEV = "mps"
W = H = 128
F = 100.0
Z = 2.0
TILE = 32


def _fixture():
    # NOTE: the kernels take ACTIVATED scales/opacities -- the trainer exps and
    # sigmoids before calling. Passing log-scales here silently produces a
    # 450px radius that trips the max-radius cull and renders nothing.
    return dict(
        quats=torch.tensor([[1.0, 0, 0, 0]], device=DEV),
        scales=torch.full((1, 3), 0.04, device=DEV),
        sh=torch.zeros(1, 16, 3, device=DEV).index_fill_(1, torch.tensor([0], device=DEV), 1.0),
        op=torch.tensor([0.99], device=DEV),
        vm=torch.eye(4, device=DEV),
        K=torch.tensor([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]], device=DEV),
    )


def _energy(u_px, backend, f):
    x = (u_px - W / 2) * Z / F
    m = torch.tensor([[x, 0.0, Z]], device=DEV, dtype=torch.float32)
    rgb, _, _ = backend(m, f["quats"], f["scales"], f["op"], f["sh"], f["K"], f["vm"],
                        W, H, tile=TILE, sh_degree=3)
    return rgb.sum().item()


@pytest.mark.parametrize("lo,hi,label", [
    (28.0, 36.0, "tile boundary u=32"),
    (-8.0, 8.0, "left screen edge u=0"),
    (120.0, 136.0, "right screen edge u=W"),
])
@pytest.mark.parametrize("backend", [metal_render, torch_ref.render],
                         ids=["metal", "torch_ref"])
def test_no_discontinuity_across_boundary(lo, hi, label, backend):
    f = _fixture()
    us = np.arange(lo, hi, 0.05)
    e = np.array([_energy(u, backend, f) for u in us])
    steps = np.abs(np.diff(e))
    med, mx = np.median(steps), steps.max()
    assert mx <= 8 * max(med, 1e-9), (
        f"{label}: step {mx:.5f} at u={us[1:][steps.argmax()]:.2f} is "
        f"{mx / max(med, 1e-9):.1f}x the median {med:.5f} -- splat dropped?"
    )


def test_metal_matches_reference_across_boundaries():
    f = _fixture()
    us = np.arange(28.0, 36.0, 0.1)
    a = np.array([_energy(u, metal_render, f) for u in us])
    b = np.array([_energy(u, torch_ref.render, f) for u in us])
    assert np.abs(a - b).max() < 1e-3


def test_exact_ellipse_filter_is_bit_exact():
    """The optional ellipse-vs-tile test must not change a single pixel.

    It drops (gaussian, tile) pairs whose conic cannot reach any pixel of the
    tile. That is exactly the rasteriser's own alpha < 1/255 cutoff expressed
    as a bound on the quadratic form, so the dropped pairs contribute nothing.
    Currently off by default (it costs more in torch than it saves) -- the test
    keeps it honest for when binning moves into a kernel.
    """
    import numpy as np
    from metal_gauss.metal_backend import _load, build_tile_lists

    torch.manual_seed(0)
    N, Wl, Hl, tl = 20000, 240, 320, 16
    g = torch.Generator(device="cpu").manual_seed(1)
    means = (torch.randn(N, 3, generator=g) * 0.5).to(DEV)
    means[:, 2] += 3.0
    quats = torch.randn(N, 4, generator=g).to(DEV)          # rotated -> elongated
    scales = (torch.rand(N, 3, generator=g) * 0.08 + 0.005).to(DEV)
    op = (torch.rand(N, generator=g) * 0.9 + 0.05).to(DEV)
    sh = (torch.randn(N, 16, 3, generator=g) * 0.3).to(DEV)
    Km = torch.tensor([[200.0, 0, Wl / 2], [0, 200.0, Hl / 2], [0, 0, 1.0]])
    vm = torch.eye(4)
    ext = _load()
    cc = (-vm[:3, :3].T @ vm[:3, 3]).contiguous()
    uv, conic, depth, rxy, valid, color = ext.preprocess_forward(
        means, quats, scales, sh, sh, op, vm.contiguous(),
        200.0, 200.0, Wl / 2, Hl / 2, Wl, Hl, 0.01, 100.0, 0.3, 1.0 * max(Wl, Hl), cc, 3)

    args = [x.contiguous() for x in (uv, conic, op, color)]
    a = build_tile_lists(uv, rxy[:, 0], depth, valid.bool(), Wl, Hl, tl, ry=rxy[:, 1])
    b = build_tile_lists(uv, rxy[:, 0], depth, valid.bool(), Wl, Hl, tl, ry=rxy[:, 1],
                         conic=conic, opacity=op)
    assert b[0].numel() < a[0].numel(), "filter removed nothing -- is it wired up?"
    rgb_a, al_a, _, _ = ext.rasterize_forward(*args, a[0], a[1], Wl, Hl, tl, a[2])
    rgb_b, al_b, _, _ = ext.rasterize_forward(*args, b[0], b[1], Wl, Hl, tl, b[2])
    assert torch.equal(rgb_a, rgb_b), f"max diff {(rgb_a - rgb_b).abs().max().item():.3e}"
    assert torch.equal(al_a, al_b)
