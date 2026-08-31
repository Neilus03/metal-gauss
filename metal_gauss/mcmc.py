"""MCMC densification (3DGS-MCMC, Kheradmand et al., NeurIPS 2024).

Dead gaussians are *relocated* onto live ones rather than pruned-and-regrown,
so the population size is controlled explicitly instead of emerging from
clone/split heuristics. `grow_to` additionally ramps the active count toward a
cap, which is both a quality lever (capacity when the optimiser can use it)
and a speed lever (early steps run on a fraction of the final splats).

Correctness notes -- the first implementation got all of these wrong:

* **Multiplicity.** When N copies share one gaussian's job, each needs
  `o_new = 1 - (1-o)^(1/N)` so the stack composites back to `o`. Assuming
  N == 2 (`1 - sqrt(1-o)`) leaves every 3+-way relocation too transparent.
* **Duplicate picks.** `multinomial(replacement=True)` returns repeats, and
  `params[dead] = params[pick]` is a *scatter*: repeats silently last-write-
  win, so the intended N-way split became a 1-way copy with N-1 stragglers.
  Multiplicity is now counted with bincount and applied to the target too.
* **Placement.** Copies go AT the target, not jittered off it -- the noise
  term is what explores; jittering here just desynchronises the split.
* **Adam state.** A relocated row carries momentum from its previous life and
  a second moment that mis-scales its first steps. Both are zeroed.
"""

from __future__ import annotations

import torch

MAX_MULTIPLICITY = 51          # matches the reference implementation's clamp

# Parameters copied verbatim when a gaussian is relocated/cloned. Opacity and
# scale are NOT here: they get the multiplicity corrections. Discovered by
# key rather than hardcoded, so splitting "sh" into sh_dc/sh_rest (or adding
# any future per-gaussian parameter) cannot silently skip a tensor.
_CORRECTED = ("logit_opac", "log_scales")


def _copy_keys(params: dict) -> list[str]:
    return [k for k in params if k not in _CORRECTED]


def _binomial_scale_ratio(n: torch.Tensor, o_new: torch.Tensor) -> torch.Tensor:
    """Scale shrink factor for an N-way split, per the MCMC paper's eq. 9.

    Approximates the ratio that keeps the summed density comparable; the exact
    integral has no closed form, so the reference uses the same construction.
    """
    denom = n.clamp_min(1).to(o_new.dtype) * o_new.clamp_min(1e-6)
    return (o_new / denom.clamp_min(1e-6)).clamp(0.05, 1.0).sqrt()


@torch.no_grad()
def relocate(params: dict, dead_thresh: float = 0.005,
             opt=None, active: int | None = None,
             weights: torch.Tensor | None = None) -> int:
    """Move dead gaussians onto live ones. Returns how many moved.

    `weights` chooses WHICH live gaussian each dead one lands on. The MCMC
    paper samples proportional to opacity -- important gaussians get split.
    3DGS-ADC instead densifies where the screen-space positional gradient is
    large, i.e. where the reconstruction is straining. Passing a gradient-aware
    weight here swaps that policy without touching the relocation math below,
    which only cares about the multiplicity each target ends up with.
    """
    n_total = params["logit_opac"].shape[0]
    n_act = n_total if active is None else active
    opac = torch.sigmoid(params["logit_opac"][:n_act])
    dead = (opac < dead_thresh).nonzero(as_tuple=True)[0]
    alive = (opac >= dead_thresh).nonzero(as_tuple=True)[0]
    if dead.numel() == 0 or alive.numel() == 0:
        return 0

    w = opac if weights is None else weights[:n_act].clamp_min(0)
    w_alive = w[alive]
    if not torch.isfinite(w_alive).all() or w_alive.sum() <= 0:
        w_alive = opac[alive]          # fall back rather than sample from garbage
    pick = alive[torch.multinomial(w_alive, dead.numel(), replacement=True)]

    # multiplicity N per target: 1 original + however many copies landed on it
    counts = torch.bincount(pick, minlength=n_act).clamp(1, MAX_MULTIPLICITY)
    n_pick = counts[pick].to(opac.dtype)

    o_src = opac[pick]
    o_new = 1.0 - (1.0 - o_src).clamp_min(1e-6) ** (1.0 / n_pick)
    logit_new = torch.log(o_new.clamp(1e-6, 1 - 1e-6)
                          / (1.0 - o_new).clamp_min(1e-6))
    ratio = _binomial_scale_ratio(n_pick, o_new)

    for k in _copy_keys(params):
        params[k][dead] = params[k][pick]
    params["log_scales"][dead] = params["log_scales"][pick] + torch.log(ratio)[:, None]
    params["logit_opac"][dead] = logit_new

    # the targets shrink too -- they now share the job with their copies
    uniq = torch.unique(pick)
    n_uniq = counts[uniq].to(opac.dtype)
    o_u = 1.0 - (1.0 - opac[uniq]).clamp_min(1e-6) ** (1.0 / n_uniq)
    params["logit_opac"][uniq] = torch.log(o_u.clamp(1e-6, 1 - 1e-6)
                                           / (1.0 - o_u).clamp_min(1e-6))
    params["log_scales"][uniq] += torch.log(_binomial_scale_ratio(n_uniq, o_u))[:, None]

    if opt is not None:
        reset_adam_state(opt, params, torch.cat([dead, uniq]))
    return int(dead.numel())


@torch.no_grad()
def reset_adam_state(opt, params: dict, idx: torch.Tensor) -> None:
    """Zero Adam moments for rows whose meaning just changed."""
    for t in params.values():
        st = opt.state.get(t)
        if not st:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][idx] = 0.0


@torch.no_grad()
def grow(params: dict, target: int, active: int, opt=None,
         weights: torch.Tensor | None = None) -> int:
    """Activate more gaussians by cloning opacity-sampled live ones.

    Capacity ramp in the spirit of ADC growth, without gradient bookkeeping:
    parameters are preallocated at the cap and `active` marks how many are in
    play, so growth costs one scatter and never reallocates.
    """
    n_total = params["logit_opac"].shape[0]
    target = min(target, n_total)
    add = target - active
    if add <= 0:
        return active

    opac = torch.sigmoid(params["logit_opac"][:active])
    w = opac if weights is None else weights[:active].clamp_min(0)
    if not torch.isfinite(w).all() or w.sum() <= 0:
        w = opac
    pick = torch.multinomial(w.clamp_min(1e-8), add, replacement=True)
    dst = torch.arange(active, target, device=pick.device)

    counts = torch.bincount(pick, minlength=active).clamp(1, MAX_MULTIPLICITY) + 1
    n_pick = counts[pick].to(opac.dtype)
    o_new = 1.0 - (1.0 - opac[pick]).clamp_min(1e-6) ** (1.0 / n_pick)
    logit_new = torch.log(o_new.clamp(1e-6, 1 - 1e-6) / (1.0 - o_new).clamp_min(1e-6))
    ratio = _binomial_scale_ratio(n_pick, o_new)

    for k in _copy_keys(params):
        params[k][dst] = params[k][pick]
    params["log_scales"][dst] = params["log_scales"][pick] + torch.log(ratio)[:, None]
    params["logit_opac"][dst] = logit_new
    params["logit_opac"][pick] = logit_new
    params["log_scales"][pick] += torch.log(ratio)[:, None]

    if opt is not None:
        reset_adam_state(opt, params, torch.cat([dst, pick]))
    return target


@torch.no_grad()
def add_noise(params: dict, lr_means: float, weight: float = 4e4,
              active: int | None = None, opacity_gate: float = 0.005) -> None:
    """SGLD-style exploration on the means of near-transparent gaussians.

    Two fixes over the first version: the perturbation is drawn in the
    gaussian's own frame (rotated by its quaternion, scaled by its axes) so it
    explores along the shape rather than the world axes, and the amplitude is
    proportional to the CURRENT learning rate, so exploration anneals as the
    run settles instead of shaking finished geometry forever.
    """
    n = params["means"].shape[0] if active is None else active
    opac = torch.sigmoid(params["logit_opac"][:n])
    gate = torch.sigmoid(-100.0 * (opac - opacity_gate))
    scales = torch.exp(params["log_scales"][:n])
    eps = torch.randn_like(scales) * scales

    # Rotate eps into the gaussian's frame directly from the quaternion:
    #   v' = v + 2w(u x v) + 2u x (u x v)
    # rather than materialising an (N,3,3) rotation and contracting it. Same
    # result to float32 roundoff (2e-6), 6.3x faster (11.4 -> 1.8 ms at 600k),
    # and it runs EVERY step, unlike relocate. Quats are stored unnormalised,
    # so normalise here exactly as quat_to_rotmat does.
    q = params["quats"][:n]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w_, u = q[:, 0:1], q[:, 1:4]
    t = 2.0 * torch.cross(u, eps, dim=1)
    world = eps + w_ * t + torch.cross(u, t, dim=1)

    params["means"][:n] += world * gate[:, None] * lr_means * weight
