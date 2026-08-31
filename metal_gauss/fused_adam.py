"""Adam in one Metal pass instead of torch's five.

torch's Adam (foreach=False, which is what torch itself selects on MPS --
foreach=True measured SLOWER here, 33.4 vs 26.5 ms) walks each tensor about
five times per step: lerp_ for the first moment, mul_ + addcmul_ for the
second, sqrt/add for the denominator, addcdiv_ for the update. Every pass
reads and writes whole buffers, so the cost is bandwidth, not arithmetic.

At this trainer's 35.4M parameters that is ~1 GB of traffic per step. Measured
Adam was 23 ms against an 8.9 ms elementwise-bandwidth floor on this machine
(111 GB/s, measured) -- 2.6x of headroom, all of it redundant passes.

State layout is deliberately identical to torch.optim.Adam's ('step',
'exp_avg', 'exp_avg_sq' in opt.state[param]) because mcmc.reset_adam_state
reaches into it to zero the moments of relocated rows. A different layout
would silently break relocation rather than fail loudly.
"""

from __future__ import annotations

import torch

from metal_gauss.metal_backend import _load


class FusedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-15):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        ext = _load()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("FusedAdam does not support sparse gradients")
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                # The kernel indexes flat; views must be contiguous to alias
                # the same storage the optimiser state was built from.
                ext.adam_step(p.view(-1), g.contiguous().view(-1),
                              st["exp_avg"].view(-1), st["exp_avg_sq"].view(-1),
                              lr, b1, b2, eps, st["step"])
        return loss
