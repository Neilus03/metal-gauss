"""Tile size at a given operating point, forward and backward.

Tile size is regime-dependent in this repo, not a constant: 32x32 is 30% faster
at 900x1600 with 2M splats and costs 35% at 270p, because 135 tiles cannot fill
the GPU. The default of 16 was chosen from the 270p regime.

The msplat per-step gap was measured at 100k splats @ 800x800, which is a third
regime nobody has swept. This measures it in-process -- same scene, same
gaussians, only `tile` differs -- because that method resolved 2.5% on the
absgrad probe, while the cross-process slope fit carries a 13% band.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metal_gauss import render                    # noqa: E402
from bench.provenance import env as _env          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaussians", type=int, default=100_000)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--tiles", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "tile_sweep.json"))
    a = ap.parse_args()

    N, W, H = a.gaussians, a.res, a.res
    torch.manual_seed(0)
    means = (torch.rand(N, 3, device="mps") * 2.0 - 1.0)
    quats = torch.randn(N, 4).to("mps")
    scales = (torch.rand(N, 3) * 0.03 + 0.005).to("mps")
    opac = (torch.rand(N) * 0.7 + 0.15).to("mps")
    sh = torch.rand(N, 1, 3).to("mps")
    rest = torch.zeros(N, 15, 3, device="mps")
    K = torch.eye(3); f = 0.8 * max(W, H)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = f, f, W / 2, H / 2
    vm = torch.eye(4)
    vm[2, 3] = 3.0

    def once(tile, grad):
        m = means.clone().requires_grad_(grad)
        rgb, _, info = render(m, quats, scales, opac, sh, K, vm, W, H,
                              backend="metal", sh_rest=rest, tile=tile)
        if grad:
            rgb.square().mean().backward()
        return info

    # 2 s sustained-load ramp: Apple Silicon boosts short bursts by up to 40%,
    # which has faked a 26% win in this repo before.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        once(16, True)
    torch.mps.synchronize()

    rows = []
    print(f"  {N:,} gaussians @ {W}x{H}\n")
    print(f"  {'tile':>5} {'fwd ms':>9} {'fwd+bwd ms':>12} {'isect':>12}")
    for tile in a.tiles:
        info = once(tile, False)
        for grad in (False, True):
            ts = []
            for i in range(a.iters + a.warm):
                torch.mps.synchronize(); t = time.perf_counter()
                once(tile, grad)
                torch.mps.synchronize()
                if i >= a.warm:
                    ts.append(time.perf_counter() - t)
            ts.sort()
            med = ts[len(ts) // 2] * 1000
            if grad:
                bwd = med
            else:
                fwd = med
        rows.append({"tile": tile, "fwd_ms": round(fwd, 3),
                     "fwd_bwd_ms": round(bwd, 3),
                     "isect": info["isect_total"], "tiles": info["tiles"]})
        print(f"  {tile:>5} {fwd:>9.2f} {bwd:>12.2f} {info['isect_total']:>12,}")

    best = min(rows, key=lambda r: r["fwd_bwd_ms"])
    cur = next((r for r in rows if r["tile"] == 16), None)
    print(f"\n  best fwd+bwd: tile {best['tile']} at {best['fwd_bwd_ms']:.2f} ms")
    if cur and best["tile"] != 16:
        print(f"  default tile 16 is {100*(cur['fwd_bwd_ms']/best['fwd_bwd_ms']-1):+.1f}% "
              f"off the best here")
    elif cur:
        print("  the default of 16 is already best at this operating point")

    Path(a.out).write_text(json.dumps(
        {"schema": 1, "env": _env(),
         "config": {"gaussians": N, "res": a.res, "iters": a.iters},
         "rows": rows}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
