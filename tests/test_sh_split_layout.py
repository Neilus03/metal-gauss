"""The kernel reads SH as one (N,16,3) tensor or as (N,1,3)+(N,15,3).

The trainer uses the split layout so Adam can hold the DC band and bands 1+ as
separate parameter groups with different learning rates; concatenating them
every step cost 11.2 ms fwd+bwd at 600k. Both layouts index the same kernel
code through a layout parameter, so the two must agree -- in value and in
gradient. A wrong stride shows up as colour drift at higher SH degrees, which
is easy to mistake for a tuning problem.

The IMAGE is asserted bit-identical (rasterize_forward uses no atomics), but
the GRADIENT cannot be: rasterize_backward accumulates through atomics, so the
fused path is not bit-reproducible against itself -- measured 1.2e-10 max
difference between two identical runs. Split-vs-fused sits at 5.8e-11, below
that floor. The tolerance here is the backend's own nondeterminism, not a
fudge factor, so it is asserted explicitly rather than with a bare allclose.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(),
                                reason="requires MPS")

N, W, H = 4000, 96, 128


def _scene(seed=0):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.randn(N, 3, generator=g).to("mps") * 0.4
    means[:, 2] += 3.0
    return dict(
        means=means,
        quats=torch.randn(N, 4, generator=g).to("mps"),
        scales=torch.rand(N, 3, generator=g).to("mps") * 0.05 + 0.01,
        opac=torch.rand(N, generator=g).to("mps") * 0.8 + 0.1,
        sh=torch.randn(N, 16, 3, generator=g).to("mps") * 0.3,
        K=torch.tensor([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]], device="mps"),
        vm=torch.eye(4, device="mps"),
    )


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_split_matches_fused(sh_degree):
    from metal_gauss.metal_backend import render
    s = _scene()

    fused_sh = s["sh"].clone().requires_grad_(True)
    rgb_f, _, _ = render(s["means"], s["quats"], s["scales"], s["opac"], fused_sh,
                         s["K"], s["vm"], W, H, sh_degree=sh_degree)
    rgb_f.square().mean().backward()

    dc = s["sh"][:, :1].clone().requires_grad_(True)
    rest = s["sh"][:, 1:].clone().requires_grad_(True)
    rgb_s, _, _ = render(s["means"], s["quats"], s["scales"], s["opac"], dc,
                         s["K"], s["vm"], W, H, sh_degree=sh_degree, sh_rest=rest)
    rgb_s.square().mean().backward()

    assert torch.equal(rgb_f, rgb_s), (
        f"images differ, max {(rgb_f - rgb_s).abs().max().item():.3e}")

    # Below the atomic-nondeterminism floor measured above (1.2e-10).
    ATOMIC_FLOOR = 1e-9
    d_dc = (fused_sh.grad[:, :1] - dc.grad).abs().max().item()
    d_rest = (fused_sh.grad[:, 1:] - rest.grad).abs().max().item()
    assert d_dc < ATOMIC_FLOOR, f"DC gradient differs by {d_dc:.3e}"
    assert d_rest < ATOMIC_FLOOR, f"bands 1+ gradient differ by {d_rest:.3e}"


def test_split_gradient_is_not_doubled():
    """Fused mode aliases d_sh and d_sh_rest to one tensor; returning it twice
    would silently double every SH gradient."""
    from metal_gauss.metal_backend import render
    s = _scene(1)
    sh = s["sh"].clone().requires_grad_(True)
    rgb, _, _ = render(s["means"], s["quats"], s["scales"], s["opac"], sh,
                       s["K"], s["vm"], W, H, sh_degree=3)
    rgb.square().mean().backward()
    g1 = sh.grad.clone()

    sh2 = s["sh"].clone().requires_grad_(True)
    rgb2, _, _ = render(s["means"], s["quats"], s["scales"], s["opac"], sh2,
                        s["K"], s["vm"], W, H, sh_degree=3)
    (2.0 * rgb2).square().mean().backward()
    # scaling the loss by 2 scales the gradient by 4, not 8
    assert torch.allclose(sh2.grad, 4.0 * g1, rtol=1e-4, atol=1e-7)


def test_full_sh_with_sh_rest_is_rejected():
    """Passing a full (N,16,3) alongside sh_rest is ambiguous -- refuse it
    rather than silently reading band 0 and ignoring the other 15."""
    from metal_gauss.metal_backend import render
    s = _scene(2)
    with pytest.raises(ValueError, match="DC band"):
        render(s["means"], s["quats"], s["scales"], s["opac"], s["sh"],
               s["K"], s["vm"], W, H, sh_degree=3, sh_rest=s["sh"][:, 1:])
