"""Per-image appearance correction for captures with auto-exposure.

A phone camera rides its auto-exposure through a walkthrough, so the same wall
is a different brightness in different training frames. Without a way to
express that, the optimiser bakes the variation into the gaussians themselves
-- as colour drift, spurious geometry in shaded regions, or opacity soup -- and
held-out views inherit the damage.

This gives each TRAINING image a small learnable photometric transform (per
channel gain + bias, optionally a full 3x3 colour matrix) applied to the render
before the loss. The gaussians then only have to explain what is actually
scene-dependent. Held-out views get the identity transform: the model must
generalise without a per-view cheat, which is what makes the held-out PSNR gain
real rather than a fitting artefact.

Same idea as gsplat's bilateral grid / NeRF-W appearance embeddings, minus the
spatial grid -- exposure and white balance are global per frame, and the global
version costs 6 parameters per image instead of thousands.
"""

from __future__ import annotations

import torch


class AppearanceModel(torch.nn.Module):
    def __init__(self, n_images: int, mode: str = "gain_bias", device: str = "mps"):
        super().__init__()
        self.mode = mode
        if mode == "gain_bias":                       # 6 params / image
            self.gain = torch.nn.Parameter(torch.ones(n_images, 3, device=device))
            self.bias = torch.nn.Parameter(torch.zeros(n_images, 3, device=device))
        elif mode == "affine":                        # 12 params / image
            eye = torch.eye(3, device=device).expand(n_images, 3, 3).clone()
            self.matrix = torch.nn.Parameter(eye)
            self.bias = torch.nn.Parameter(torch.zeros(n_images, 3, device=device))
        else:
            raise ValueError(f"unknown appearance mode {mode!r}")

    def forward(self, rgb: torch.Tensor, idx: int) -> torch.Tensor:
        """rgb: (H,W,3) render -> photometrically corrected render."""
        if self.mode == "gain_bias":
            return rgb * self.gain[idx] + self.bias[idx]
        return rgb @ self.matrix[idx].T + self.bias[idx]

    def regulariser(self) -> torch.Tensor:
        """Keep transforms near identity so they correct exposure, not content."""
        if self.mode == "gain_bias":
            return ((self.gain - 1.0) ** 2).mean() + (self.bias ** 2).mean()
        eye = torch.eye(3, device=self.matrix.device)
        return ((self.matrix - eye) ** 2).mean() + (self.bias ** 2).mean()
