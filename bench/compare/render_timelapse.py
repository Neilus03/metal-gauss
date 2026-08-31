"""Render every checkpoint from ONE held-out camera and score it.

Produces the frames for the side-by-side convergence animation: a single
camera, three implementations, indexed by wall-clock. Same camera for all three
-- comparing different viewpoints would be meaningless, and it is the kind of
thing that is easy to get wrong silently, so the camera index is recorded in
the output.

Frames are JPEG at 400px to keep the published artefact small; the metrics are
computed on the full-precision render before encoding, not on the JPEG.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.provenance import env as _env          # noqa: E402
from metal_gauss import render                    # noqa: E402
from metal_gauss.blender import load_blender      # noqa: E402
from metal_gauss.io import load_ply               # noqa: E402
from metal_gauss.train import _gaussian_kernel    # noqa: E402
from metal_gauss.train import ssim as ssim_metal  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*",
                    default=[str(ROOT / "bench" / "results" / "timelapse.json"),
                             str(ROOT / "bench" / "results" / "timelapse_msplat.json")])
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--view", type=int, default=0,
                    help="held-out camera shown in the panels")
    ap.add_argument("--metric-views", type=int, default=20,
                    help="held-out views averaged for the PSNR/SSIM curve. The "
                         "panel shows ONE camera because that is what a viewer "
                         "wants to watch, but a single view's PSNR is noisy "
                         "enough to contradict the benchmark -- on view 0 of "
                         "lego it makes our curve appear to DEGRADE while the "
                         "200-view score rises monotonically. The curve must "
                         "track the benchmark, so it is averaged.")
    ap.add_argument("--res", type=int, default=400,
                    help="resolution of the DISPLAYED panel image only")
    ap.add_argument("--metric-res", type=int, default=800,
                    help="resolution the curve is scored at. Must be the "
                         "TRAINING resolution: scoring at 400px while the "
                         "models trained at 800 made our curve appear to "
                         "plateau at 27.2 dB, because a denser model aliases "
                         "more at reduced resolution -- the same effect "
                         "--antialias exists to fix. At 800px the same two "
                         "checkpoints go 29.57 -> 30.53, matching the "
                         "benchmark.")
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "timelapse_frames.json"))
    a = ap.parse_args()

    from PIL import Image
    scene = load_blender(a.scene, a.res)          # display resolution
    mscene = load_blender(a.scene, a.metric_res)  # scoring resolution
    views = scene.heldout
    v = views[a.view]
    H, W = v.image.shape[:2]
    kernel = _gaussian_kernel(device="mps")
    # evenly spaced across the held-out set rather than the first N, which
    # would sample one region of the camera orbit
    mv_all = mscene.heldout
    step_v = max(1, len(mv_all) // a.metric_views)
    mviews = mv_all[::step_v][:a.metric_views]
    print(f"  panel camera: {getattr(v, 'name', '?')};  "
          f"metrics averaged over {len(mviews)} held-out views\n")

    rows, done = [], {}
    for src in a.inputs:
        p = Path(src)
        if not p.exists():
            print(f"  {p.name}: MISSING, skipped"); continue
        d = json.loads(p.read_text())
        for r in d["rows"]:
            if "done_at" in r:
                done[r["impl"]] = r["done_at"]; continue
            ply = Path(r["ply"])
            if not ply.exists():
                continue
            try:
                sp = load_ply(str(ply), device="mps")
                ps, ss_l, im = [], [], None
                for k, mv in enumerate(mviews):
                    mh, mw = mv.image.shape[:2]
                    with torch.no_grad():
                        rgb, _, _ = render(sp.means, sp.quats, sp.scales,
                                           sp.opacities, sp.sh, mv.K, mv.viewmat,
                                           mw, mh, sh_degree=3, backend="metal",
                                           background=(1.0, 1.0, 1.0))
                    # NOT `r`: that is the checkpoint row in the enclosing
                    # loop, and shadowing it here made this crash on the row
                    # lookup several lines later
                    im_v = rgb.clamp(0, 1)
                    g = mv.image.to("mps").float() / 255.0
                    ps.append(float((-10.0 * torch.log10(
                        ((im_v - g) ** 2).mean().clamp_min(1e-12))).item()))
                    ss_l.append(float(ssim_metal(im_v, g, kernel).item()))
                # the panel image is the chosen camera, rendered separately
                with torch.no_grad():
                    rgb, _, _ = render(sp.means, sp.quats, sp.scales, sp.opacities,
                                       sp.sh, v.K, v.viewmat, W, H, sh_degree=3,
                                       backend="metal", background=(1.0, 1.0, 1.0))
                im = rgb.clamp(0, 1)
                psnr = float(np.mean(ps))
                ss = float(np.mean(ss_l))
            except Exception as e:
                print(f"  {ply.name}: FAILED {type(e).__name__}: {e}")
                continue
            arr = (im.cpu().numpy() * 255).astype(np.uint8)
            buf = Path("/tmp/_frame.jpg")
            Image.fromarray(arr).save(buf, quality=a.quality)
            b64 = base64.b64encode(buf.read_bytes()).decode("ascii")
            rows.append({"impl": r["impl"], "t": r["t"], "psnr": round(psnr, 3),
                         "ssim": round(ss, 5), "n": int(len(sp)), "jpg": b64})
            print(f"  {r['impl']:<12} t={r['t']:7.1f}s  PSNR {psnr:6.2f}  "
                  f"SSIM {ss:.4f}  {len(sp):>7,} splats", flush=True)

    rows.sort(key=lambda r: (r["impl"], r["t"]))
    gtb = Path("/tmp/_gt.jpg")
    Image.fromarray(v.image.numpy()).save(gtb, quality=90)
    out = {"schema": 1, "env": _env(),
           "config": {"scene": Path(a.scene).name, "view_index": a.view,
                      "view_name": getattr(v, "name", "?"), "res": a.res,
                      "metric_views": len(mviews), "metric_res": a.metric_res},
           "done_at": done,
           "gt_jpg": base64.b64encode(gtb.read_bytes()).decode("ascii"),
           "frames": rows}
    Path(a.out).write_text(json.dumps(out))
    mb = Path(a.out).stat().st_size / 1e6
    print(f"\n  {len(rows)} frames, {mb:.1f} MB -> {a.out}")


if __name__ == "__main__":
    main()
