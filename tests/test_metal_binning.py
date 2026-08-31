"""Metal binning must render exactly what torch binning renders.

The Metal path also applies the exact ellipse-vs-tile test, so it produces a
SHORTER intersection list -- roughly 38% shorter on real scenes. That is the
point, and it is only legitimate because the dropped pairs cannot reach the
rasteriser's own alpha >= 1/255 cutoff anywhere in the tile. So the list length
is allowed to differ; the rendered image is not.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(),
                                reason="requires MPS")

DEV = "mps"


def _scene(N=20000, W=240, H=320, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.randn(N, 3, generator=g) * 0.5).to(DEV)
    means[:, 2] += 3.0
    return dict(
        means=means,
        quats=torch.randn(N, 4, generator=g).to(DEV),        # rotated -> elongated
        scales=(torch.rand(N, 3, generator=g) * 0.08 + 0.005).to(DEV),
        op=(torch.rand(N, generator=g) * 0.9 + 0.05).to(DEV),
        sh=(torch.randn(N, 16, 3, generator=g) * 0.3).to(DEV),
        W=W, H=H,
    )


@pytest.mark.parametrize("tile", [16, 32])
def test_metal_binning_renders_identically(tile):
    from metal_gauss.metal_backend import (_load, build_tile_lists,
                                           build_tile_lists_metal)
    s = _scene()
    W, H = s["W"], s["H"]
    ext = _load()
    vm = torch.eye(4)
    cc = (-vm[:3, :3].T @ vm[:3, 3]).contiguous()
    f = 200.0
    uv, conic, depth, rxy, valid, color = ext.preprocess_forward(
        s["means"], s["quats"], s["scales"], s["sh"], s["sh"], s["op"],
        vm.contiguous(), f, f, W / 2, H / 2, W, H, 0.01, 100.0, 0.3,
        1.0 * max(W, H), cc, 3)

    a = build_tile_lists(uv, rxy[:, 0], depth, valid.bool(), W, H, tile, ry=rxy[:, 1])
    b = build_tile_lists_metal(uv, rxy, depth, valid, conic, s["op"], W, H, tile)

    assert b[0].numel() < a[0].numel(), "ellipse test removed nothing"
    assert a[2] == b[2] and a[1].shape == b[1].shape

    args = [x.contiguous() for x in (uv, conic, s["op"], color)]
    ra, aa, _, _ = ext.rasterize_forward(*args, a[0], a[1], W, H, tile, a[2])
    rb, ab, _, _ = ext.rasterize_forward(*args, b[0], b[1], W, H, tile, b[2])
    assert torch.equal(ra, rb), f"image differs, max {(ra - rb).abs().max().item():.3e}"
    assert torch.equal(aa, ab)


def test_offsets_are_monotonic_and_complete():
    from metal_gauss.metal_backend import _load, build_tile_lists_metal
    s = _scene(seed=2)
    W, H, tile = s["W"], s["H"], 16
    ext = _load()
    vm = torch.eye(4)
    cc = (-vm[:3, :3].T @ vm[:3, 3]).contiguous()
    uv, conic, depth, rxy, valid, _ = ext.preprocess_forward(
        s["means"], s["quats"], s["scales"], s["sh"], s["sh"], s["op"],
        vm.contiguous(), 200.0, 200.0, W / 2, H / 2, W, H, 0.01, 100.0, 0.3,
        1.0 * max(W, H), cc, 3)
    gid, off, tx = build_tile_lists_metal(uv, rxy, depth, valid, conic, s["op"],
                                          W, H, tile)
    o = off.to(torch.int64)
    assert int(o[0]) == 0
    assert int(o[-1]) == gid.numel()
    assert bool((o[1:] >= o[:-1]).all())


def test_empty_scene_is_handled():
    """All gaussians invalid: the count pass must produce zero work, not a
    zero-length allocation the write pass then indexes into."""
    from metal_gauss.metal_backend import _load, build_tile_lists_metal
    ext = _load()
    N, W, H = 128, 64, 64
    means = torch.zeros(N, 3, device=DEV)
    means[:, 2] = -5.0                       # entirely behind the camera
    quats = torch.zeros(N, 4, device=DEV); quats[:, 0] = 1
    scales = torch.full((N, 3), 0.01, device=DEV)
    op = torch.full((N,), 0.5, device=DEV)
    sh = torch.zeros(N, 16, 3, device=DEV)
    vm = torch.eye(4)
    cc = (-vm[:3, :3].T @ vm[:3, 3]).contiguous()
    uv, conic, depth, rxy, valid, _ = ext.preprocess_forward(
        means, quats, scales, sh, sh, op, vm.contiguous(), 100.0, 100.0,
        W / 2, H / 2, W, H, 0.01, 100.0, 0.3, 1.0 * max(W, H), cc, 3)
    gid, off, tx = build_tile_lists_metal(uv, rxy, depth, valid, conic, op, W, H, 16)
    assert gid.numel() == 0
    assert int(off[-1]) == 0
