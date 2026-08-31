"""Reference 3D Gaussian Splatting rasterizer in pure PyTorch.

Runs on MPS and CPU. Fully differentiable end to end -- no custom autograd
functions, no CUDA, no compiled extension. It exists for two reasons:

  1. Stage 8 needs to re-render splats from a held-out pose on a Mac, and no
     shipped differentiable MPS rasterizer exists.
  2. It is the correctness oracle for the Metal kernels. A hand-written Metal
     rasterizer with nothing to check it against is a random number generator.

It is deliberately slow and deliberately simple. The one non-obvious trick is
that alpha compositing is expressed as an exclusive cumulative product rather
than a sequential loop, which keeps it both vectorised and differentiable:

    T_i = prod_{j<i} (1 - a_j)          <- exclusive cumprod over depth
    C   = sum_i  c_i * a_i * T_i

Tiling exists to bound memory, not for speed: the per-(tile, gaussian, pixel)
alpha tensor is the memory wall, so tiles are processed in chunks.
"""

from __future__ import annotations

import math

import torch

from metal_gauss.sh import eval_sh, num_sh_bases


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3). Normalised internally, as 3DGS stores unnormalised."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def build_cov3d(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Sigma = R S S^T R^T, from (N,4) rotation and (N,3) scale."""
    R = quat_to_rotmat(quats)
    M = R * scales[:, None, :]          # R @ diag(s)
    return M @ M.transpose(1, 2)


def project(
    means: torch.Tensor, cov3d: torch.Tensor, viewmat: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float, W: int, H: int,
    near: float, far: float, blur: float = 0.3, max_radius_frac: float = 1.0,
    antialias: bool = False,
):
    """World -> camera -> screen, with the EWA 2D covariance.

    Returns (uv, conic, depth, radius, valid, opacity_scale).

    `opacity_scale` is 1 unless `antialias` is set, in which case it carries the
    Mip-Splatting / gsplat "antialiased" compensation described below. It is
    always returned so callers have one code path.
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p_cam = means @ R.T + t
    z = p_cam[:, 2]

    valid = (z > near) & (z < far)
    zc = z.clamp_min(near)

    u = fx * p_cam[:, 0] / zc + cx
    v = fy * p_cam[:, 1] / zc + cy
    uv = torch.stack([u, v], dim=-1)

    # Clamp the point at which the Jacobian is evaluated to just outside the
    # frustum. The perspective Jacobian carries a 1/z^2 term, so a Gaussian
    # that is near the camera and far off-axis produces an essentially
    # unbounded 2D covariance. Without this guard a real scene rendered here
    # showed a maximum projected radius of 1.4e8 pixels: single Gaussians
    # covering every tile, sorting first because they are nearest, and
    # saturating the entire image to flat grey.
    #
    # The linearisation is only valid near the projection centre anyway, so
    # clamping it is the correct fix rather than a workaround. This matches the
    # limx/limy clamp used by the reference CUDA implementations.
    lim_x = 1.3 * (0.5 * W / fx)
    lim_y = 1.3 * (0.5 * H / fy)
    tx = torch.clamp(p_cam[:, 0] / zc, -lim_x, lim_x) * zc
    ty = torch.clamp(p_cam[:, 1] / zc, -lim_y, lim_y) * zc

    # Jacobian of the perspective projection at each Gaussian centre.
    zero = torch.zeros_like(zc)
    J = torch.stack([
        torch.stack([fx / zc, zero, -fx * tx / (zc * zc)], dim=-1),
        torch.stack([zero, fy / zc, -fy * ty / (zc * zc)], dim=-1),
    ], dim=1)                                              # (N,2,3)

    T = J @ R                                              # (N,2,3)
    cov2d = T @ cov3d @ T.transpose(1, 2)                  # (N,2,2)

    # Low-pass filter: guarantees every Gaussian covers at least ~one pixel,
    # otherwise sub-pixel Gaussians alias violently and their gradients vanish.
    #
    # Dilating the covariance without touching opacity spreads the same total
    # alpha over a larger footprint, so a sub-pixel Gaussian is both blurred AND
    # dimmed. That is the erosion Mip-Splatting identifies, and it bites hardest
    # when render resolution differs from training resolution -- which is our
    # DEFAULT since --num-downscales 2 trains at 1/4 -> 1/2 -> full.
    #
    # The fix (Mip-Splatting's 2D Mip filter, and gsplat's "antialiased" mode)
    # is to rescale opacity by the square root of the determinant ratio, so the
    # dilation preserves total energy instead of losing it:
    #
    #     opacity' = opacity * sqrt(det(cov2d) / det(cov2d + blur*I))
    #
    # It is <= 1 always, and -> 1 for Gaussians already much wider than a pixel,
    # so it only touches the ones that were being eroded.
    det_before = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0])
    cov2d = cov2d + blur * torch.eye(2, device=cov2d.device, dtype=cov2d.dtype)

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    if antialias:
        ratio = det_before.clamp_min(0.0) / det.clamp_min(1e-12)
        opacity_scale = ratio.clamp(0.0, 1.0).sqrt()
    else:
        opacity_scale = torch.ones_like(det)
    valid = valid & (det > 1e-12)
    det_safe = det.clamp_min(1e-12)
    conic = torch.stack([c / det_safe, -b / det_safe, a / det_safe], dim=-1)

    # 3-sigma extent from the larger eigenvalue of the 2D covariance.
    mid = 0.5 * (a + c)
    disc = (mid * mid - det_safe).clamp_min(0).sqrt()
    lam = (mid + disc).clamp_min(1e-12)
    radius = 3.0 * lam.sqrt()

    # A Gaussian wider than the image cannot be carrying real detail, and one
    # that covers every tile destroys the per-tile depth budget for everything
    # behind it. Drop them rather than let them dominate.
    max_radius = max_radius_frac * max(W, H)
    valid = valid & (radius < max_radius)

    valid = valid & (u + radius > 0) & (u - radius < W) & (v + radius > 0) & (v - radius < H)
    return uv, conic, z, radius, valid, opacity_scale


def _tile_bins(uv, radius, valid, depth, W, H, tile, max_per_tile, device):
    """Assign Gaussians to tiles and keep the nearest `max_per_tile` in each.

    Returns (n_tiles_x, n_tiles_y, buf) where buf is (n_tiles, max_per_tile)
    holding Gaussian indices, -1 for empty slots.
    """
    tx, ty = math.ceil(W / tile), math.ceil(H / tile)
    n_tiles = tx * ty

    idx = torch.nonzero(valid, as_tuple=True)[0]
    if idx.numel() == 0:
        return tx, ty, torch.full((n_tiles, 1), -1, dtype=torch.long, device=device), 0, 0

    u, v, r = uv[idx, 0], uv[idx, 1], radius[idx]
    x0 = ((u - r) / tile).floor().clamp(0, tx - 1).long()
    x1 = ((u + r) / tile).floor().clamp(0, tx - 1).long()
    y0 = ((v - r) / tile).floor().clamp(0, ty - 1).long()
    y1 = ((v + r) / tile).floor().clamp(0, ty - 1).long()

    nx, ny = (x1 - x0 + 1), (y1 - y0 + 1)
    counts = nx * ny
    total = int(counts.sum().item())

    # Expand each Gaussian into one entry per tile it touches.
    g = torch.repeat_interleave(torch.arange(idx.numel(), device=device), counts)
    start = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=device) - start[g]
    dx, dy = within % nx[g], within // nx[g]
    tile_id = (y0[g] + dy) * tx + (x0[g] + dx)

    # Sort by depth, then stable-sort by tile: within each tile, front to back.
    d = depth[idx][g]
    o1 = torch.argsort(d)
    o2 = torch.argsort(tile_id[o1], stable=True)
    order = o1[o2]
    tile_sorted, gauss_sorted = tile_id[order], idx[g][order]

    # Rank within tile, so we can keep only the nearest max_per_tile.
    tcounts = torch.bincount(tile_sorted, minlength=n_tiles)
    tstart = torch.cumsum(tcounts, 0) - tcounts
    rank = torch.arange(tile_sorted.numel(), device=device) - tstart[tile_sorted]

    # Size the buffer to the busiest tile rather than a fixed budget, so
    # nothing is dropped unless the safety cap actually bites.
    needed = int(tcounts.max().item())
    width = min(needed, max_per_tile)
    keep = rank < width
    buf = torch.full((n_tiles, width), -1, dtype=torch.long, device=device)
    buf[tile_sorted[keep], rank[keep]] = gauss_sorted[keep]
    return tx, ty, buf, int((~keep).sum().item()), total


def rasterize(
    uv, conic, opacity, color, depth, radius, valid, W, H,
    tile: int = 16, max_per_tile: int = 8192, tile_chunk: int = 32,
    slab: int = 256, background=None, min_transmittance: float = 1e-4,
):
    """Alpha-composite front to back. Returns (rgb (H,W,3), alpha (H,W), info).

    Compositing runs over depth-sorted *slabs* rather than a single fixed-size
    per-tile budget, carrying transmittance across slab boundaries:

        weight_i = alpha_i * T_in * prod_{j<i within slab} (1 - alpha_j)
        T_out    = T_in * prod_{i in slab} (1 - alpha_i)

    This is exact and keeps peak memory at (tiles x slab x pixels) regardless of
    how many Gaussians land in a tile. The earlier version instead kept only the
    nearest `max_per_tile` per tile, which silently discarded 89% of tile
    intersections on a real 1M-Gaussian scene and rendered a near-empty image.
    `max_per_tile` remains only as a safety cap and is reported when it bites.
    """
    device, dtype = uv.device, uv.dtype
    tx, ty, buf, dropped, total = _tile_bins(
        uv, radius, valid, depth, W, H, tile, max_per_tile, device)
    n_tiles = tx * ty
    P = tile * tile

    rgb = torch.zeros(n_tiles, P, 3, device=device, dtype=dtype)
    acc = torch.zeros(n_tiles, P, device=device, dtype=dtype)

    py, px = torch.meshgrid(
        torch.arange(tile, device=device, dtype=dtype),
        torch.arange(tile, device=device, dtype=dtype),
        indexing="ij",
    )
    off = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1) + 0.5   # (P,2)
    tile_ix = torch.arange(n_tiles, device=device)
    origin = torch.stack([(tile_ix % tx) * tile, (tile_ix // tx) * tile], dim=-1).to(dtype)

    K = buf.shape[1]
    for lo in range(0, n_tiles, tile_chunk):
        hi = min(lo + tile_chunk, n_tiles)
        b = buf[lo:hi]
        occupied = int((b >= 0).sum().item())
        if occupied == 0:
            continue
        C = b.shape[0]
        pix = origin[lo:hi, None, :] + off[None, :, :]        # (C,P,2)

        T = torch.ones(C, P, device=device, dtype=dtype)
        out_rgb = torch.zeros(C, P, 3, device=device, dtype=dtype)

        for s0 in range(0, K, slab):
            s1 = min(s0 + slab, K)
            bs = b[:, s0:s1]
            if not bool((bs >= 0).any()):
                break                                        # tiles are packed
            if float(T.max()) < min_transmittance:
                break                                        # fully opaque already
            mask = bs >= 0
            g = bs.clamp_min(0)

            d = pix[:, None, :, :] - uv[g][:, :, None, :]    # (C,S,P,2)
            cg = conic[g]
            power = -0.5 * (
                cg[..., 0:1] * d[..., 0] * d[..., 0]
                + 2.0 * cg[..., 1:2] * d[..., 0] * d[..., 1]
                + cg[..., 2:3] * d[..., 1] * d[..., 1]
            )
            alpha = (opacity[g][..., None] * power.exp()).clamp(max=0.999)
            alpha = torch.where(mask[..., None], alpha, torch.zeros_like(alpha))
            alpha = torch.where(alpha > 1.0 / 255.0, alpha, torch.zeros_like(alpha))

            one_minus = 1.0 - alpha
            cp = torch.cumprod(one_minus, dim=1)
            excl = cp / one_minus.clamp_min(1e-10)           # exclusive prefix
            w = alpha * excl * T[:, None, :]                 # carry transmittance in
            out_rgb = out_rgb + torch.einsum("csp,csj->cpj", w, color[g])
            acc[lo:hi] = acc[lo:hi] + w.sum(dim=1)
            T = T * cp[:, -1]                                # transmittance out

        rgb[lo:hi] = out_rgb

    img = rgb.reshape(ty, tx, tile, tile, 3).permute(0, 2, 1, 3, 4).reshape(ty * tile, tx * tile, 3)
    alp = acc.reshape(ty, tx, tile, tile).permute(0, 2, 1, 3).reshape(ty * tile, tx * tile)
    img, alp = img[:H, :W], alp[:H, :W]

    if background is not None:
        bg = torch.as_tensor(background, device=device, dtype=dtype)
        img = img + (1.0 - alp).clamp_min(0)[..., None] * bg

    return img, alp, {
        "tiles": n_tiles,
        "isect_total": total,
        "isect_dropped": dropped,
        "isect_dropped_frac": round(dropped / total, 5) if total else 0.0,
        "max_gaussians_in_a_tile": K,
        "visible_gaussians": int(valid.sum().item()),
    }


def render(
    means, quats, scales, opacities, sh, K, viewmat, W, H,
    sh_degree: int = 3, tile: int = 32, max_per_tile: int = 8192,
    tile_chunk: int = 32, near: float = 0.01, far: float = 100.0, slab: int = 256,
    background=(0.0, 0.0, 0.0), colors=None, max_radius_frac: float = 1.0,
    antialias: bool = False,
):
    """Render Gaussians from one camera.

    means      (N,3) world space
    quats      (N,4) wxyz, need not be normalised
    scales     (N,3) linear (not log)
    opacities  (N,)  in [0,1] (not logit)
    sh         (N,B,3) SH coefficients; ignored if `colors` is given
    K          (3,3) pinhole intrinsics
    viewmat    (4,4) world -> camera
    """
    device = means.device
    fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()

    cov3d = build_cov3d(quats, scales)
    uv, conic, depth, radius, valid, opacity_scale = project(
        means, cov3d, viewmat, fx, fy, cx, cy, W, H, near, far,
        max_radius_frac=max_radius_frac, antialias=antialias,
    )
    # The compensation multiplies opacity, so it must be applied before
    # compositing rather than folded into the conic.
    opacities = opacities * opacity_scale

    if colors is None:
        cam_center = -viewmat[:3, :3].T @ viewmat[:3, 3]
        dirs = means - cam_center
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        colors = (eval_sh(sh_degree, sh[:, : num_sh_bases(sh_degree)], dirs) + 0.5).clamp_min(0.0)

    return rasterize(
        uv, conic, opacities, colors, depth, radius, valid, W, H,
        tile=tile, max_per_tile=max_per_tile, tile_chunk=tile_chunk,
        slab=slab, background=background,
    )
