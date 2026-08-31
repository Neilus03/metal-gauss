"""Correctness tests for the reference rasterizer.

These are the oracle the Metal kernels will be judged against, so they check
behaviour that must hold rather than pinning current numbers.
"""

from __future__ import annotations

import math

import pytest
import torch

from metal_gauss import render

DEV = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def intrinsics(W, H, f=None):
    f = f or 0.8 * max(W, H)
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    return K


def one_gaussian(device, dtype=torch.float32, z=3.0, scale=0.15, opacity=1.0, color=(1, 0, 0)):
    means = torch.tensor([[0.0, 0.0, z]], device=device, dtype=dtype)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    scales = torch.full((1, 3), scale, device=device, dtype=dtype)
    opac = torch.tensor([opacity], device=device, dtype=dtype)
    cols = torch.tensor([color], device=device, dtype=dtype)
    return means, quats, scales, opac, cols


@pytest.mark.parametrize("device", DEV)
def test_single_gaussian_lands_at_principal_point(device):
    W = H = 64
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    m, q, s, o, c = one_gaussian(device)

    rgb, alpha, info = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")

    assert info["visible_gaussians"] == 1
    peak = torch.nonzero(alpha == alpha.max())[0]
    # Centred Gaussian on the optical axis must image at the principal point.
    assert abs(peak[0].item() - H / 2) <= 1
    assert abs(peak[1].item() - W / 2) <= 1
    assert alpha.max() > 0.9
    assert alpha[0, 0] < 1e-3          # falls off to nothing at the corner
    assert rgb[H // 2, W // 2, 0] > 0.9  # red channel
    assert rgb[H // 2, W // 2, 2] < 1e-3


@pytest.mark.parametrize("device", DEV)
def test_depth_ordering_front_occludes_back(device):
    """A red Gaussian in front of a blue one must render red, not blue."""
    W = H = 32
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    means = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]], device=device)  # blue far, red near
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, device=device)
    scales = torch.full((2, 3), 0.3, device=device)
    opac = torch.tensor([1.0, 1.0], device=device)
    cols = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=device)

    rgb, alpha, _ = render(means, quats, scales, opac, None, K, viewmat, W, H,
                           colors=cols, backend="torch_ref")
    centre = rgb[H // 2, W // 2]
    assert centre[0] > 0.9, f"expected near/red to win, got {centre.tolist()}"
    assert centre[2] < 0.1


@pytest.mark.parametrize("device", DEV)
def test_behind_camera_is_culled(device):
    W = H = 32
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    m, q, s, o, c = one_gaussian(device, z=-3.0)
    rgb, alpha, info = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")
    assert info["visible_gaussians"] == 0
    assert alpha.max() == 0


@pytest.mark.parametrize("device", DEV)
def test_alpha_is_exactly_linear_in_opacity(device):
    """alpha = opacity * exp(power), so alpha/opacity must be constant.

    Note peak alpha is NOT equal to opacity even at opacity 1.0: a Gaussian
    centred on the optical axis projects to (cx, cy), but pixels are sampled at
    their centres (+0.5), so the nearest sample sits 0.5px off-peak and picks up
    the Gaussian falloff. Asserting peak == opacity would be asserting a bug.
    """
    W = H = 32
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    peaks = []
    for op in (0.25, 0.5, 1.0):
        m, q, s, o, c = one_gaussian(device, opacity=op)
        _, alpha, _ = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")
        peaks.append(alpha.max().item())

    assert peaks[0] < peaks[1] < peaks[2]
    ratios = [p / o for p, o in zip(peaks, (0.25, 0.5, 1.0))]
    assert max(ratios) - min(ratios) < 1e-5, f"not linear in opacity: {ratios}"


@pytest.mark.parametrize("device", DEV)
def test_alpha_saturates_at_the_clamp(device):
    """A Gaussian wide enough that falloff is negligible must hit the 0.999 clamp.

    Kept comfortably inside the max-radius cull: the point is to test alpha
    saturation, not to smuggle in an oversized Gaussian.
    """
    W = H = 128
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    m, q, s, o, c = one_gaussian(device, scale=0.5, opacity=1.0)
    _, alpha, _ = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")
    assert alpha.max().item() == pytest.approx(0.999, abs=1e-3)


@pytest.mark.parametrize("device", DEV)
def test_farther_gaussian_is_smaller(device):
    W = H = 64
    K = intrinsics(W, H).to(device)
    viewmat = torch.eye(4, device=device)
    areas = []
    for z in (2.0, 4.0):
        m, q, s, o, c = one_gaussian(device, z=z)
        _, alpha, _ = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")
        areas.append((alpha > 0.05).sum().item())
    # Twice the distance -> roughly a quarter of the projected area.
    assert areas[0] > areas[1] * 2.5, f"areas {areas}"


@pytest.mark.parametrize("device", DEV)
def test_translating_camera_moves_the_splat(device):
    W = H = 64
    K = intrinsics(W, H).to(device)
    m, q, s, o, c = one_gaussian(device)
    viewmat = torch.eye(4, device=device)
    viewmat[0, 3] = 0.5                      # shift camera in +x
    _, alpha, _ = render(m, q, s, o, None, K, viewmat, W, H, colors=c, backend="torch_ref")
    peak = torch.nonzero(alpha == alpha.max())[0]
    assert peak[1].item() > W / 2 + 2        # image moves right


def test_gradients_flow_to_every_input():
    """Every learnable parameter must receive a finite, non-zero gradient."""
    W = H = 32
    K = intrinsics(W, H)
    viewmat = torch.eye(4)
    means = torch.tensor([[0.05, -0.02, 3.0]], requires_grad=True)
    quats = torch.tensor([[1.0, 0.1, 0.0, 0.0]], requires_grad=True)
    scales = torch.tensor([[0.2, 0.15, 0.18]], requires_grad=True)
    opac = torch.tensor([0.7], requires_grad=True)
    sh = (torch.randn(1, 16, 3) * 0.3).requires_grad_(True)

    rgb, alpha, _ = render(means, quats, scales, opac, sh, K, viewmat, W, H,
                           sh_degree=3, backend="torch_ref")
    rgb.square().mean().backward()

    for name, t in [("means", means), ("quats", quats), ("scales", scales),
                    ("opacities", opac), ("sh", sh)]:
        assert t.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient has non-finite values"
        assert t.grad.abs().sum() > 0, f"{name} gradient is identically zero"


@pytest.mark.skipif("mps" not in DEV, reason="no MPS device")
def test_mps_matches_cpu():
    """MPS and CPU must agree; if they do not, every MPS number is suspect."""
    torch.manual_seed(0)
    W, H, N = 96, 64, 300
    K = intrinsics(W, H)
    viewmat = torch.eye(4)

    means = torch.randn(N, 3) * 0.6 + torch.tensor([0.0, 0.0, 4.0])
    quats = torch.randn(N, 4)
    scales = torch.rand(N, 3) * 0.1 + 0.02
    opac = torch.rand(N) * 0.8 + 0.1
    cols = torch.rand(N, 3)

    out_cpu = render(means, quats, scales, opac, None, K, viewmat, W, H,
                     colors=cols, backend="torch_ref")
    args_mps = [t.to("mps") for t in (means, quats, scales, opac, cols)]
    out_mps = render(args_mps[0], args_mps[1], args_mps[2], args_mps[3], None,
                     K.to("mps"), viewmat.to("mps"), W, H,
                     colors=args_mps[4], backend="torch_ref")

    for i, name in [(0, "rgb"), (1, "alpha")]:
        a, b = out_cpu[i], out_mps[i].cpu()
        assert torch.allclose(a, b, atol=2e-4), \
            f"{name} max abs diff {(a - b).abs().max().item():.3e}"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        render(None, None, None, None, None, None, None, 8, 8, backend="cuda")


def test_metal_backend_never_silently_falls_back():
    """Asking for metal must either use metal or raise -- never quietly use torch.

    Two failure paths, and the test accepts either depending on whether the
    extension is built on this machine: an unbuilt extension must say so, and a
    built one must reject non-MPS input rather than migrating it. What is
    forbidden in both cases is returning a torch_ref render under the name
    'metal', which would make every backend comparison meaningless.
    """
    with pytest.raises(RuntimeError, match="Refusing to silently substitute|requires MPS tensors"):
        render(torch.zeros(1, 3), torch.zeros(1, 4), torch.ones(1, 3), torch.ones(1),
               None, torch.eye(3), torch.eye(4), 8, 8,
               colors=torch.ones(1, 3), backend="metal")


def test_gradcheck_float64():
    """Analytic gradients must match finite differences in float64.

    This is the strongest correctness statement available for the reference
    implementation, and it is what makes the reference usable as an oracle for
    the Metal backward kernels later.

    Parameters are chosen to stay away from the two non-smooth points in the
    rasterizer -- the 0.999 alpha clamp and the 1/255 contribution cutoff --
    since gradcheck legitimately fails at a kink.
    """
    torch.manual_seed(7)
    W = H = 12
    K = intrinsics(W, H).double()
    viewmat = torch.eye(4, dtype=torch.float64)

    means = torch.tensor([[0.03, -0.05, 3.0], [-0.12, 0.08, 2.6]], dtype=torch.float64)
    quats = torch.tensor([[1.0, 0.05, -0.02, 0.03], [0.98, -0.1, 0.05, 0.0]], dtype=torch.float64)
    scales = torch.tensor([[0.22, 0.19, 0.2], [0.17, 0.21, 0.18]], dtype=torch.float64)
    opac = torch.tensor([0.55, 0.4], dtype=torch.float64)
    cols = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.8]], dtype=torch.float64)

    for t in (means, quats, scales, opac, cols):
        t.requires_grad_(True)

    def f(means_, quats_, scales_, opac_, cols_):
        rgb, alpha, _ = render(means_, quats_, scales_, opac_, None, K, viewmat, W, H,
                               colors=cols_, backend="torch_ref")
        return rgb

    assert torch.autograd.gradcheck(
        f, (means, quats, scales, opac, cols), eps=1e-6, atol=1e-6, rtol=1e-4,
        nondet_tol=0.0,
    )


@pytest.mark.parametrize("device", DEV)
def test_near_offaxis_gaussian_does_not_explode(device):
    """A near, far-off-axis Gaussian must not project to an unbounded radius.

    The perspective Jacobian has a 1/z^2 term. Without clamping the point at
    which it is evaluated, a Gaussian close to the camera and well off-axis
    produces an essentially infinite 2D covariance. On a real reconstruction
    this reached a projected radius of 1.4e8 pixels; those Gaussians covered
    every tile, sorted first because they were nearest, and rendered the whole
    image flat grey. Regression test for that.
    """
    from metal_gauss.torch_ref import build_cov3d, project

    W = H = 128
    K = intrinsics(W, H).to(device)
    fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
    viewmat = torch.eye(4, device=device)

    # Close to the camera and far outside the frustum.
    means = torch.tensor([[6.0, -4.0, 0.02], [0.0, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, device=device)
    scales = torch.full((2, 3), 0.1, device=device)

    cov3d = build_cov3d(quats, scales)
    _, _, _, radius, valid, _ = project(means, cov3d, viewmat, fx, fy, cx, cy,
                                        W, H, 0.01, 100.0)
    assert torch.isfinite(radius).all(), "projected radius must stay finite"
    assert not valid[0], "a Gaussian this close and off-axis must be rejected"
    assert valid[1], "the well-behaved Gaussian must survive"
    assert radius[valid].max() <= max(W, H), \
        f"kept radius {radius[valid].max().item():.1f}px exceeds the image"
