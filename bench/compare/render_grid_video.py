"""Square 2x2 convergence video, for feeds that punish wide aspect ratios.

The row layout in render_timelapse_gif.py is 2.4:1. That is fine in a README and
poor almost everywhere else: social feeds are mobile-first and letterbox video
renders small, so the same four lanes go into a 2x2 grid at 1:1 instead.

MP4 rather than GIF: H.264 at 1080x1080 is a fraction of the size of an
equivalent GIF and is not re-encoded on upload.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "bench" / "results" / "timelapse_frames.json"
LANES = [("metal-gauss", "metal-gauss", (217, 185, 78)),
         ("brush", "Brush", (188, 159, 184)),
         ("msplat", "msplat", (127, 190, 203)),
         ("spirula", "spirula-studio", (143, 184, 148))]
BG, INK, DIM = (17, 20, 25), (233, 230, 223), (140, 147, 158)


def _font(size, bold=False):
    for base in ("/System/Library/Fonts/", "/Library/Fonts/"):
        p = Path(base) / "HelveticaNeue.ttc"
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size, index=1 if bold else 0)
            except OSError:
                pass
    print("    [warn] no TrueType face; falling back to bitmap")
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument("--out", default=str(ROOT / "assets" / "timelapse_square.mp4"))
    a = ap.parse_args()

    data = json.loads(SRC.read_text())
    lanes = []
    for key, label, col in LANES:
        fr = sorted((f for f in data["frames"] if f["impl"] == key), key=lambda f: f["t"])
        for f in fr:
            f["img"] = Image.open(io.BytesIO(base64.b64decode(f["jpg"]))).convert("RGB")
        lanes.append({"label": label, "col": col, "frames": fr})
    T = max(l["frames"][-1]["t"] for l in lanes)

    S = a.size
    head = int(S * 0.105)                 # title strip
    cell = (S - head) // 2
    f_title, f_clock = _font(int(S * 0.038), True), _font(int(S * 0.046), True)
    f_lane, f_num, f_lab = _font(int(S * 0.026), True), _font(int(S * 0.030), True), _font(int(S * 0.016))

    tmp = Path(tempfile.mkdtemp())
    n_hold = int(a.hold * a.fps)
    for i in range(a.frames + n_hold):
        t = min(i, a.frames - 1) / (a.frames - 1) * T
        im = Image.new("RGB", (S, S), BG)
        d = ImageDraw.Draw(im)
        d.text((int(S * 0.028), int(S * 0.030)), "Same scene, same 6.5 minutes",
               font=f_title, fill=INK)
        clock = f"{t:5.1f}s"
        d.text((S - int(S * 0.028) - d.textlength(clock, font=f_clock), int(S * 0.026)),
               clock, font=f_clock, fill=INK)

        for j, ln in enumerate(lanes):
            ox, oy = (j % 2) * cell, head + (j // 2) * cell
            cur = None
            for f in ln["frames"]:
                if f["t"] <= t:
                    cur = f
            pad = int(S * 0.012)
            box = cell - 2 * pad
            img_h = box - int(S * 0.072)
            if cur is None:
                d.rectangle([ox + pad, oy + pad, ox + pad + box, oy + pad + img_h], fill=(11, 14, 18))
            else:
                im.paste(cur["img"].resize((box, img_h), Image.LANCZOS), (ox + pad, oy + pad))
            d.rectangle([ox + pad, oy + pad, ox + pad + box, oy + pad + 4], fill=ln["col"])
            ty = oy + pad + img_h + int(S * 0.008)
            d.text((ox + pad + 4, ty), ln["label"], font=f_lane, fill=ln["col"])
            if cur:
                s_psnr = f"{cur['psnr']:.2f} dB"
                d.text((ox + pad + box - 4 - d.textlength(s_psnr, font=f_num), ty - int(S * 0.004)),
                       s_psnr, font=f_num, fill=INK)
        im.save(tmp / f"f{i:04d}.png")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps),
                    "-i", str(tmp / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    print(f"  {a.frames}+{n_hold} frames at {S}x{S} -> {out} "
          f"({out.stat().st_size/1048576:.2f} MB, {(a.frames+n_hold)/a.fps:.1f}s)")


if __name__ == "__main__":
    main()
