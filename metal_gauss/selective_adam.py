"""Index-based Adam that only updates visible gaussians.

Per view only ~20-40% of gaussians project into the frustum; the rest have
exactly-zero gradients, yet a dense Adam still reads and writes all N x 64
parameter/state floats every step. Updating the visible subset by index cuts
optimizer time and memory traffic proportionally (gsplat's selective_adam
idea). Bias correction uses a PER-GAUSSIAN step count, matching what dense
Adam would have done had the invisible steps simply not occurred.
"""

from __future__ import annotations

import torch


class SelectiveAdam:
    def __init__(self, groups: list[dict], eps: float = 1e-15,
                 betas: tuple[float, float] = (0.9, 0.999)):
        self.groups = groups
        self.eps = eps
        self.b1, self.b2 = betas
        self.state = {}
        for g in groups:
            for t in g["params"]:
                self.state[id(t)] = {
                    "m": torch.zeros_like(t),
                    "v": torch.zeros_like(t),
                    "steps": torch.zeros(t.shape[0], device=t.device),
                }

    @torch.no_grad()
    def step(self, visible: torch.Tensor) -> None:
        # Index-based updates pay ~15 gather/scatter launches; when most
        # gaussians are visible (early training, wide-angle views) that costs
        # more than it saves. Above 50% visibility, run the same per-gaussian-
        # step math as full-width masked ops instead -- identical result,
        # dense-op speed. The index path wins once opacity culling and the
        # frustum shrink visibility (late training, big budgets).
        n = visible.numel()
        nvis = int(visible.sum())
        if nvis > n // 2:
            self._step_masked(visible)
            return
        idx = visible.nonzero(as_tuple=True)[0]
        for g in self.groups:
            lr = g["lr"]
            for t in g["params"]:
                if t.grad is None:
                    continue
                st = self.state[id(t)]
                st["steps"][idx] += 1
                gsel = t.grad[idx]
                m = st["m"][idx].mul_(self.b1).add_(gsel, alpha=1 - self.b1)
                v = st["v"][idx].mul_(self.b2).addcmul_(gsel, gsel, value=1 - self.b2)
                st["m"][idx] = m
                st["v"][idx] = v
                k = st["steps"][idx]
                bc1 = 1 - self.b1 ** k
                bc2 = 1 - self.b2 ** k
                shape = [-1] + [1] * (t.dim() - 1)
                step_size = lr * (bc2.sqrt() / bc1).reshape(shape)
                t[idx] -= step_size * m / (v.sqrt() + self.eps)

    def zero_grad(self) -> None:
        for g in self.groups:
            for t in g["params"]:
                t.grad = None

    @torch.no_grad()
    def _step_masked(self, visible: torch.Tensor) -> None:
        for g in self.groups:
            lr = g["lr"]
            for t in g["params"]:
                if t.grad is None:
                    continue
                st = self.state[id(t)]
                shape = [-1] + [1] * (t.dim() - 1)
                m_mask = visible.reshape(shape).to(t.dtype)
                st["steps"] += visible.to(st["steps"].dtype)
                grad = t.grad * m_mask
                # visible rows: standard Adam accumulate; invisible: unchanged
                st["m"] = torch.where(m_mask.bool(), st["m"] * self.b1 + grad * (1 - self.b1), st["m"])
                st["v"] = torch.where(m_mask.bool(), st["v"] * self.b2 + grad * grad * (1 - self.b2), st["v"])
                k = st["steps"].clamp_min(1.0)
                bc1 = 1 - self.b1 ** k
                bc2 = 1 - self.b2 ** k
                step_size = (lr * bc2.sqrt() / bc1).reshape(shape)
                t -= m_mask * step_size * st["m"] / (st["v"].sqrt() + self.eps)
