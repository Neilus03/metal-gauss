"""Real spherical harmonics evaluation, degrees 0-3.

Matches the basis and coefficient ordering used by 3DGS .ply files, so
Gaussians exported by Brush can be evaluated here without reshuffling.
"""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)


def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh: (N, K, 3) coefficients, dirs: (N, 3) unit view directions -> (N, 3) RGB.

    Returns radiance *before* the +0.5 offset and clamp that 3DGS applies, so
    callers can decide; `render` does it.
    """
    k = num_sh_bases(degree)
    if sh.shape[1] < k:
        raise ValueError(f"need {k} SH bases for degree {degree}, got {sh.shape[1]}")

    out = C0 * sh[:, 0]
    if degree == 0:
        return out

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if degree == 1:
        return out

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = (out
           + C2[0] * xy * sh[:, 4]
           + C2[1] * yz * sh[:, 5]
           + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
           + C2[3] * xz * sh[:, 7]
           + C2[4] * (xx - yy) * sh[:, 8])
    if degree == 2:
        return out

    out = (out
           + C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
           + C3[1] * xy * z * sh[:, 10]
           + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
           + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
           + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
           + C3[5] * z * (xx - yy) * sh[:, 14]
           + C3[6] * x * (xx - 3.0 * yy) * sh[:, 15])
    return out
