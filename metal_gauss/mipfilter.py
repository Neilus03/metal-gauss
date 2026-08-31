"""Mip-Splatting's 3D smoothing filter.

The 2D Mip filter (`--antialias`) is a screen-space quantity: it rescales
opacity by sqrt(det ratio) at RENDER time, so it cannot be baked into an
exported ply, which is the only reason it is not the default despite being
worth +6.7 dB at 200px render resolution.

The 3D filter is different. It band-limits each Gaussian to the maximal
sampling rate of the training views that actually see it, and that is a
property of the Gaussian, not of the camera it is being drawn from. So it CAN
be folded into the exported scales and opacity, and any viewer renders it
correctly with no cooperation.

    filter_3d = min_over_cameras(depth) / max_over_cameras(focal) * sqrt(0.2)
    scales'   = sqrt(scales^2 + filter_3d^2)
    opacity'  = opacity * sqrt(prod(scales^2) / prod(scales^2 + filter_3d^2))

The opacity term conserves energy exactly as the 2D filter's does: widening a
Gaussian without dimming it would brighten the render.

Both functions are differentiable, so nothing here needs a hand-written
adjoint.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_3d_filter(means: torch.Tensor, views, near: float = 0.2,
                      margin: float = 0.15) -> torch.Tensor:
    """Per-Gaussian low-pass scale, one value per Gaussian.

    `views` need `.K` and `.viewmat` and an `.image` for the frame size. The
    pass is O(N * V) and is meant to be recomputed periodically, not per step:
    the Gaussians move under MCMC relocation, so a filter computed once at
    init goes stale, but recomputing it every step is pure waste.

    A Gaussian seen by no camera gets the largest distance any Gaussian got,
    which is the conservative choice -- it is filtered the most, so it cannot
    introduce aliasing it was never checked for.
    """
    dev = means.device
    n = means.shape[0]
    dist = torch.full((n,), float("inf"), device=dev)
    seen = torch.zeros(n, dtype=torch.bool, device=dev)
    focal = 0.0

    for v in views:
        vm = v.viewmat.to(dev) if v.viewmat.device != dev else v.viewmat
        K = v.K.to(dev) if v.K.device != dev else v.K
        R, t = vm[:3, :3], vm[:3, 3]
        pc = means @ R.T + t
        z = pc[:, 2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        H, W = v.image.shape[:2]
        zc = z.clamp_min(1e-6)
        u = fx * pc[:, 0] / zc + cx
        w = fy * pc[:, 1] / zc + cy
        # a margin beyond the frame: a Gaussian just outside still contributes
        # to edge pixels, and excluding it would leave the border unfiltered
        inview = ((z > near)
                  & (u > -margin * W) & (u < (1.0 + margin) * W)
                  & (w > -margin * H) & (w < (1.0 + margin) * H))
        dist = torch.where(inview, torch.minimum(dist, z), dist)
        seen |= inview
        focal = max(focal, fx, fy)

    if seen.any():
        dist[~seen] = dist[seen].max()
    else:
        dist.fill_(1.0)
    if focal <= 0.0:
        focal = 1.0
    return (dist / focal) * (0.2 ** 0.5)


def apply_3d_filter(scales: torch.Tensor, opacities: torch.Tensor,
                    filter_3d: torch.Tensor):
    """Widen the Gaussians and dim them to match. Returns (scales, opacities).

    Differentiable in both inputs; `filter_3d` is treated as a constant, which
    is what Mip-Splatting does -- it is a property of the camera set, not a
    parameter being fitted.
    """
    f2 = (filter_3d ** 2).unsqueeze(-1)
    s2 = scales ** 2
    s2f = s2 + f2
    new_scales = torch.sqrt(s2f)
    # sqrt(det(S^2) / det(S^2 + f^2 I)) for a diagonal S
    coef = torch.sqrt((s2.prod(dim=-1) / s2f.prod(dim=-1)).clamp(0.0, 1.0))
    return new_scales, opacities * coef
