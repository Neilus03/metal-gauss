"""Single render interface, swappable backends.

Stage 8 must never block on kernel work, so it always calls through here.
`torch_ref` is the differentiable pure-PyTorch reference; `metal` is the
custom kernel path, which is validated against `torch_ref` before it is
allowed to be used for anything.
"""

from __future__ import annotations

BACKENDS = ("torch_ref", "metal")


def render(*args, backend: str = "torch_ref", **kwargs):
    if backend == "torch_ref":
        from metal_gauss import torch_ref
        return torch_ref.render(*args, **kwargs)
    if backend == "metal":
        try:
            from metal_gauss import metal_backend
        except ImportError as e:
            raise RuntimeError(
                "metal backend not built. Build it, or pass backend='torch_ref'. "
                "Refusing to silently substitute a different renderer."
            ) from e
        return metal_backend.render(*args, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
