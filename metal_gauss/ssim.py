"""SSIM with an analytic gradient. Reference math for csrc/ssim.metal.

Status: mathematically exact (value bit-identical to autograd, gradient cosine
1.00000000, rel err 6e-7) but MEASURED SLOWER in torch: 69 ms vs 43 ms for
loss fwd+bwd at 900x1600. The closed form replaces autograd's convolution
backward with a 9-group blur plus a chain of elementwise ops, and each of
those ops materialises a full-resolution temporary -- which costs more than
what it saves. torch's own conv backward is simply well optimised.

Kept because the hard part is done and validated: this is the reference math
for a FUSED METAL kernel, where every elementwise term stays in registers and
the temporaries disappear. That is where the win actually lives.

UPDATE: ported. `csrc/ssim.metal` fuses the elementwise tail (the convolutions
stay in torch, which does them well), giving 69.2 -> 38.4 ms fwd+bwd at
900x1600 with the gradient cosine still 1.00000000. The prediction above held;
this file remains the readable derivation the kernel was written from.

Every blur here is the same separable 11-tap gaussian, and a gaussian blur is
self-adjoint (symmetric kernel), so the adjoint pass reuses the identical
operator -- which is what makes the closed form cheap.

Notation, all per-pixel after blurring with window w:
    mx = w*x,  my = w*y
    sxx = w*x^2 - mx^2,  syy = w*y^2 - my^2,  sxy = w*xy - mx*my
    S = (A*B)/(C*D),  A = 2 mx my + C1,  B = 2 sxy + C2,
                      C = mx^2 + my^2 + C1,  D = sxx + syy + C2
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def gaussian_kernel(size: int = 11, sigma: float = 1.5, device="mps", groups: int = 1):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).view(1, 1, 1, size).expand(groups, 1, 1, size).contiguous()


def _blur(t: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """Separable blur; `t` is (1,G,H,W) and k1 is shaped for G groups."""
    g = t.shape[1]
    k = k1[:g]
    pad = k.shape[-1] // 2
    t = F.conv2d(t, k, padding=(0, pad), groups=g)
    return F.conv2d(t, k.transpose(2, 3), padding=(pad, 0), groups=g)


class _SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, k15, k9):
        stack = torch.cat([x, y, x * x, y * y, x * y], dim=1)
        b = _blur(stack, k15)
        mx, my, exx, eyy, exy = b.split(3, dim=1)
        sxx = exx - mx * mx
        syy = eyy - my * my
        sxy = exy - mx * my

        A = 2 * mx * my + C1
        B = 2 * sxy + C2
        C = mx * mx + my * my + C1
        D = sxx + syy + C2
        S = (A * B) / (C * D)
        ctx.save_for_backward(x, y, mx, my, A, B, C, D, k9)
        return S.mean()

    @staticmethod
    def backward(ctx, grad_out):
        x, y, mx, my, A, B, C, D, k9 = ctx.saved_tensors
        n = x.numel()
        g = grad_out / n                       # d(mean)/d(S_i)

        CD = C * D
        # partials of S w.r.t. the four aggregates
        dS_dA = g * B / CD
        dS_dB = g * A / CD
        dS_dC = -g * A * B / (CD * C)
        dS_dD = -g * A * B / (CD * D)

        # ... and of the aggregates w.r.t. the blurred moments.
        # A = 2 mx my + C1 ; C = mx^2 + my^2 + C1
        # B = 2(w*xy - mx my) + C2 ; D = (w*x^2 - mx^2) + (w*y^2 - my^2) + C2
        d_wxy = 2 * dS_dB                       # B via w*xy
        d_wxx = dS_dD                           # D via w*x^2
        d_mx = (2 * my * dS_dA + 2 * mx * dS_dC
                - 2 * my * dS_dB                # B's -2 mx my term
                - 2 * mx * dS_dD)               # D's -mx^2 term

        # Adjoint of the blur is the blur itself (symmetric separable kernel).
        blurred = _blur(torch.cat([d_mx, d_wxx, d_wxy], dim=1), k9)
        b_mx, b_wxx, b_wxy = blurred.split(3, dim=1)

        # chain through the pointwise products inside the blur arguments
        dx = b_mx + 2 * x * b_wxx + y * b_wxy
        return dx, None, None, None


def ssim(a: torch.Tensor, b: torch.Tensor, k15: torch.Tensor,
         k9: torch.Tensor) -> torch.Tensor:
    """a, b: (H,W,3) in [0,1]. Differentiable w.r.t. `a` only (b is the target)."""
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    return _SSIM.apply(x, y, k15, k9)
