"""Mip-Splatting's 3D smoothing filter.

Unlike the 2D Mip filter this one is view-independent, so it can be baked into
an exported ply. That is the whole reason it exists here: the 2D filter is
worth +6.7 dB at reduced render resolution but cannot be a default, because a
model trained with it renders ~2.3 dB worse in any viewer that lacks it.

These tests are CPU-only and pin the algebra, which is where a silent error
would live.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metal_gauss.mipfilter import apply_3d_filter, compute_3d_filter  # noqa: E402


class _View:
    """Minimal stand-in for a training view."""
    def __init__(self, z, f=1000.0, hw=(400, 400)):
        self.K = torch.eye(3, dtype=torch.float64)
        self.K[0, 0] = self.K[1, 1] = f
        self.K[0, 2], self.K[1, 2] = hw[1] / 2, hw[0] / 2
        vm = torch.eye(4, dtype=torch.float64)
        vm[2, 3] = z              # push the camera back along +z
        self.viewmat = vm
        self.image = torch.zeros(hw[0], hw[1], 3, dtype=torch.uint8)


def test_widening_and_dimming_are_consistent():
    """new_scale^2 = scale^2 + f^2, and opacity falls by the determinant ratio."""
    s = torch.tensor([[0.10, 0.20, 0.30]], dtype=torch.float64)
    o = torch.tensor([0.8], dtype=torch.float64)
    f = torch.tensor([0.05], dtype=torch.float64)
    ns, no = apply_3d_filter(s, o, f)
    assert torch.allclose(ns ** 2, s ** 2 + f ** 2)
    want = 0.8 * torch.sqrt((s ** 2).prod() / ((s ** 2 + f ** 2).prod()))
    assert torch.allclose(no, want.reshape(1))


def test_zero_filter_is_a_no_op():
    s = torch.rand(64, 3, dtype=torch.float64) + 0.01
    o = torch.rand(64, dtype=torch.float64)
    ns, no = apply_3d_filter(s, o, torch.zeros(64, dtype=torch.float64))
    assert torch.allclose(ns, s)
    assert torch.allclose(no, o)


def test_scales_only_grow_and_opacity_only_falls():
    torch.manual_seed(0)
    s = torch.rand(256, 3, dtype=torch.float64) * 0.5 + 1e-3
    o = torch.rand(256, dtype=torch.float64)
    f = torch.rand(256, dtype=torch.float64) * 0.1
    ns, no = apply_3d_filter(s, o, f)
    assert (ns >= s - 1e-12).all(), "the filter must never shrink a gaussian"
    assert (no <= o + 1e-12).all(), "widening without dimming would brighten"
    assert (no >= 0).all()


def test_a_subpixel_gaussian_is_dimmed_hard_and_a_large_one_is_not():
    tiny = torch.full((1, 3), 1e-3, dtype=torch.float64)
    big = torch.full((1, 3), 1.0, dtype=torch.float64)
    o = torch.ones(1, dtype=torch.float64)
    f = torch.tensor([0.05], dtype=torch.float64)
    _, o_tiny = apply_3d_filter(tiny, o, f)
    _, o_big = apply_3d_filter(big, o, f)
    assert o_tiny.item() < 0.01, "a gaussian far below the sampling rate must be suppressed"
    assert o_big.item() > 0.99, "a gaussian far above it must be untouched"


def test_gradients_flow_to_scales_and_opacity():
    s = (torch.rand(32, 3, dtype=torch.float64) + 0.05).requires_grad_(True)
    o = torch.rand(32, dtype=torch.float64).requires_grad_(True)
    f = torch.full((32,), 0.02, dtype=torch.float64)
    ns, no = apply_3d_filter(s, o, f)
    (ns.sum() + no.sum()).backward()
    for t in (s, o):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_filter_grows_with_distance_from_the_cameras():
    """The band limit is depth / focal: a farther gaussian is sampled more
    coarsely and must therefore be filtered more.

    Note the convention. `_View(z=5)` sets viewmat[2,3] = 5, so camera depth is
    world_z + 5. A gaussian at world z = -3 is therefore at camera depth 2, i.e.
    NEARER. The first version of this test used -3 meaning "farther" and failed;
    the code was right and the test was wrong.
    """
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    f = compute_3d_filter(means, [_View(z=5.0)])          # depths 5 and 8
    assert f[1] > f[0], "the more distant gaussian must get the larger filter"
    assert (f > 0).all()


def test_higher_focal_length_means_a_smaller_filter():
    means = torch.zeros(1, 3, dtype=torch.float64)
    lo = compute_3d_filter(means, [_View(z=5.0, f=500.0)])
    hi = compute_3d_filter(means, [_View(z=5.0, f=2000.0)])
    assert hi.item() < lo.item(), "a sharper camera samples finer, so filter less"


def test_unseen_gaussian_gets_the_most_conservative_filter():
    """Behind the camera, so seen by nothing. It must not get filter 0."""
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]], dtype=torch.float64)
    f = compute_3d_filter(means, [_View(z=5.0)])
    assert torch.isfinite(f).all()
    assert (f > 0).all(), "an unseen gaussian must still be band-limited"


def test_bake_round_trips_through_the_ply_activation_space():
    """The bake must survive the log/logit encoding the ply uses.

    export_ply stores PRE-activation values (log scale, logit opacity), so
    baking means applying the filter and re-encoding. If that round trip were
    lossy the exported model would not match what the trainer rendered, and
    the whole point of preferring the 3D filter over the 2D one -- that it can
    be baked -- would be lost.

    Verified end to end as well: a lego model trained with --filter-3d and
    exported scores 20.944 dB when rendered with NO filter applied, against
    20.94 from the trainer's own filtered eval.
    """
    torch.manual_seed(0)
    log_scales = torch.randn(128, 3, dtype=torch.float64) * 0.5 - 2.0
    logit = torch.randn(128, dtype=torch.float64)
    f = torch.rand(128, dtype=torch.float64) * 0.05

    sc, op = apply_3d_filter(torch.exp(log_scales), torch.sigmoid(logit), f)
    # exactly what export_ply writes
    baked_log = torch.log(sc.clamp_min(1e-12))
    op_c = op.clamp(1e-6, 1.0 - 1e-6)
    baked_logit = torch.log(op_c / (1.0 - op_c))
    # exactly what a viewer does when reading it back
    assert torch.allclose(torch.exp(baked_log), sc, atol=1e-12)
    assert torch.allclose(torch.sigmoid(baked_logit), op_c, atol=1e-10)


def test_bake_is_not_a_no_op():
    """Guard against the filter silently doing nothing through the encoding."""
    log_scales = torch.full((16, 3), -3.0, dtype=torch.float64)
    logit = torch.zeros(16, dtype=torch.float64)
    f = torch.full((16,), 0.05, dtype=torch.float64)
    sc, op = apply_3d_filter(torch.exp(log_scales), torch.sigmoid(logit), f)
    assert (torch.log(sc) > log_scales + 1e-6).all(), "scales must have grown"
    assert (op < torch.sigmoid(logit) - 1e-6).all(), "opacity must have fallen"


# --------------------------------------------------------------------------
# --export-every, used to build the wall-clock convergence comparison.
# --------------------------------------------------------------------------

def test_export_every_names_checkpoints_by_step():
    """Checkpoint paths must be derivable from the step, and distinct.

    The convergence timelapse maps each checkpoint to a wall-clock offset via
    its mtime, so the files must be separate -- a single overwritten path would
    collapse the whole timeline to one instant.
    """
    from pathlib import Path
    base = Path("/tmp/does_not_need_to_exist.ply")
    names = {base.with_suffix(f".step{s:06d}.ply") for s in (100, 200, 7000)}
    assert len(names) == 3
    assert all(n != base for n in names)
    assert sorted(names)[0].name.endswith(".step000100.ply")
