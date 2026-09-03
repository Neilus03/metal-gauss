"""Load/save 3D Gaussian Splatting .ply files.

The 3DGS .ply convention stores *parameterised* values, not the values the
rasteriser wants: opacity is a logit, scale is a log, and the quaternion is
unnormalised. Reading it naively -- which is easy to do, since every field has
a plausible-looking numeric range -- gives a scene that renders as fog.

SH layout follows the INRIA reference: f_dc_{0,1,2} is the degree-0 term per
channel, and f_rest_{0..44} is (3 channels x 15 higher bases) stored
channel-major, i.e. f_rest_[c*15 + (b-1)].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Splats:
    means: torch.Tensor      # (N,3) world
    quats: torch.Tensor      # (N,4) wxyz, normalised
    scales: torch.Tensor     # (N,3) linear
    opacities: torch.Tensor  # (N,)  in [0,1]
    sh: torch.Tensor         # (N,B,3)
    sh_degree: int

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device) -> "Splats":
        return Splats(
            self.means.to(device), self.quats.to(device), self.scales.to(device),
            self.opacities.to(device), self.sh.to(device), self.sh_degree,
        )

    def subset(self, idx) -> "Splats":
        return Splats(self.means[idx], self.quats[idx], self.scales[idx],
                      self.opacities[idx], self.sh[idx], self.sh_degree)


def load_ply(path: str | Path, device: str = "cpu", dtype=torch.float32) -> Splats:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    names = {p.name for p in v.properties}

    def col(n):
        return np.asarray(v[n], dtype=np.float64)

    means = np.stack([col("x"), col("y"), col("z")], axis=1)
    quats = np.stack([col(f"rot_{i}") for i in range(4)], axis=1)   # wxyz
    scales = np.exp(np.stack([col(f"scale_{i}") for i in range(3)], axis=1))
    opac = 1.0 / (1.0 + np.exp(-col("opacity")))                    # sigmoid

    n_rest = len([n for n in names if n.startswith("f_rest_")])
    # `// 3` on its own truncates: 46 coefficients would read as 45, silently
    # dropping a column and shifting the channel-major layout underneath it.
    # The whole-degree check below catches many such counts but not all.
    if n_rest % 3:
        raise ValueError(f"{path}: {n_rest} f_rest coefficients is not a whole "
                         f"number per channel")
    per_channel = n_rest // 3
    n_bases = per_channel + 1
    degree = int(round(n_bases ** 0.5)) - 1
    if (degree + 1) ** 2 != n_bases:
        raise ValueError(f"{path}: {n_rest} f_rest coefficients is not a whole SH degree")

    sh = np.zeros((len(means), n_bases, 3), np.float64)
    for c in range(3):
        sh[:, 0, c] = col(f"f_dc_{c}")
        for b in range(per_channel):
            sh[:, b + 1, c] = col(f"f_rest_{c * per_channel + b}")

    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)

    t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    return Splats(t(means), t(quats), t(scales), t(opac), t(sh), degree)
