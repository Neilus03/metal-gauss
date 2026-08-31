"""Sequential convergence reel: each scene optimises in turn, full frame.

A grid shows eight scenes at one ninth the size each and asks the viewer to
track all of them. A reel gives every scene the whole frame and one job: watch
this blur become an object, then the next. Same information, no legend.

Two passes, because they need different resources: render (GPU, one image per
checkpoint) then compose (CPU, ffmpeg). Rendering is cached on disk so the
composition can be re-cut without re-rendering.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCENES = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]


def render_pass(index: Path, cache: Path, res: int, spin: float,
                metric_views: int, metric_res: int) -> None:
    """Rotating display frame, plus the VALIDATION PSNR for each checkpoint.

    Two different things, deliberately kept apart:

      * the image rotates along the official test orbit, so the viewer can see
        it is a 3D scene rather than a photo sharpening;
      * the number is the validation PSNR -- the mean over a fixed set of
        held-out views at TRAINING resolution -- which is the same quantity the
        benchmark tables report.

    Scoring against the single rotating view instead would make the number swing
    with camera angle (some views are simply harder) and would not match any
    published figure. Metric resolution must be the training resolution: scoring
    below it penalises exactly the checkpoints carrying the most splats.
    """
    import torch
    from PIL import Image
    from metal_gauss.blender import load_blender
    from metal_gauss.io import load_ply
    from metal_gauss import render
    from bench.runner import require_gpu_exclusive
    require_gpu_exclusive()

    def psnr_of(sp, v, dev="mps"):
        H, W = v.image.shape[:2]
        with torch.no_grad():
            rgb, _, _ = render(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
                               v.K, v.viewmat, W, H, sh_degree=3, backend="metal",
                               background=(1.0, 1.0, 1.0))
        rgb = rgb.clamp(0, 1)
        gt = v.image.to(rgb.device).float() / 255.0
        return rgb, float((-10 * torch.log10(((rgb - gt) ** 2).mean().clamp_min(1e-12))).item())

    data = json.loads(index.read_text())
    for sc, d in data.items():
        out = cache / sc
        out.mkdir(parents=True, exist_ok=True)
        meta_p = out / "psnr.json"
        if meta_p.exists() and len(list(out.glob("*.jpg"))) == len(d["rows"]):
            print(f"  {sc}: cached", flush=True)
            continue
        disp = load_blender(str(ROOT / "data/nerf_synthetic" / sc), res)
        met = load_blender(str(ROOT / "data/nerf_synthetic" / sc), metric_res)
        step = max(1, len(met.heldout) // metric_views)
        mviews = met.heldout[::step][:metric_views]
        views, n = disp.heldout, len(d["rows"])
        psnrs = []
        for i, row in enumerate(d["rows"]):
            sp = load_ply(row["ply"], device="mps")
            v = views[int((i / max(1, n - 1)) * spin * (len(views) - 1)) % len(views)]
            rgb, _ = psnr_of(sp, v)
            Image.fromarray((rgb * 255).to(torch.uint8).cpu().numpy()).save(
                out / f"{i:04d}.jpg", quality=92)
            psnrs.append(sum(psnr_of(sp, mv)[1] for mv in mviews) / len(mviews))
        meta_p.write_text(json.dumps(psnrs))
        print(f"  {sc}: {n} frames, validation PSNR {psnrs[0]:.1f} -> {psnrs[-1]:.1f} dB",
              flush=True)


def compose(index: Path, cache: Path, out: Path, size: int, fps: int,
            per_scene: float, hold: float) -> None:
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    def font(px, bold=True):
        for base in ("/System/Library/Fonts/", "/Library/Fonts/"):
            f = Path(base) / "HelveticaNeue.ttc"
            if f.exists():
                try:
                    return ImageFont.truetype(str(f), px, index=1 if bold else 0)
                except OSError:
                    pass
        return ImageFont.load_default()

    data = json.loads(index.read_text())
    f_main = font(int(size * 0.036))
    f_lab = font(int(size * 0.020), bold=False)
    tmp = Path(tempfile.mkdtemp())
    n = 0
    n_per, n_hold = int(per_scene * fps), int(hold * fps)
    for sc in SCENES:
        if sc not in data:
            continue
        rows = data[sc]["rows"]
        files = sorted((cache / sc).glob("*.jpg"))
        pj = cache / sc / "psnr.json"
        psnrs = json.loads(pj.read_text()) if pj.exists() else []
        if not files:
            continue
        for k in range(n_per + n_hold):
            frac = min(k, n_per - 1) / (n_per - 1)
            idx = min(int(frac * (len(files) - 1)), len(files) - 1)
            im = Image.open(files[idx]).convert("RGB").resize((size, size), Image.LANCZOS)
            d = ImageDraw.Draw(im)
            strip = int(size * 0.105)
            d.rectangle([0, size - strip, size, size], fill=(14, 17, 21))
            pad, base = int(size * 0.032), size - strip + int(size * 0.030)
            d.text((pad, base), sc, font=f_main, fill=(233, 230, 223))
            t = rows[min(idx, len(rows) - 1)]["t"]
            mid = f"{t:.0f}s"
            d.text(((size - d.textlength(mid, font=f_main)) / 2, base), mid,
                   font=f_main, fill=(233, 230, 223))
            if idx < len(psnrs):
                q = f"{psnrs[idx]:.1f} dB"
                d.text((size - pad - d.textlength(q, font=f_main), base), q,
                       font=f_main, fill=(217, 185, 78))
            im.save(tmp / f"f{n:05d}.png")
            n += 1
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(out)], check=True)
    print(f"  {n} frames -> {out} ({out.stat().st_size/1048576:.2f} MB, {n/fps:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/tmp/scenes8/index.json")
    ap.add_argument("--cache", default="/tmp/scenes8/frames")
    ap.add_argument("--out", default=str(ROOT / "assets" / "scene_reel.mp4"))
    ap.add_argument("--res", type=int, default=600, help="render resolution")
    ap.add_argument("--size", type=int, default=1080, help="video size")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--per-scene", type=float, default=1.9)
    ap.add_argument("--hold", type=float, default=0.45)
    ap.add_argument("--metric-views", type=int, default=10)
    ap.add_argument("--metric-res", type=int, default=800)
    ap.add_argument("--spin", type=float, default=0.55,
                    help="fraction of the test orbit swept per scene")
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    index, cache = Path(a.index), Path(a.cache)
    if not a.skip_render:
        render_pass(index, cache, a.res, a.spin, a.metric_views, a.metric_res)
    compose(index, cache, Path(a.out), a.size, a.fps, a.per_scene, a.hold)


if __name__ == "__main__":
    main()
