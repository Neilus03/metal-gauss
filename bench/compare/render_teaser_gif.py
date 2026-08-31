"""Square hover-teaser GIF: one lane converging, for a 160x160 slot.

The four-panel comparison GIF is 860x358. Dropped into a square thumbnail with
object-fit: cover it crops to roughly one sixth of one panel, which shows
nothing. This renders a single lane square instead, so the thing a reader sees
on hover is the actual point: a blur resolving into a bulldozer, against a
running clock.
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


def _font(size, bold=False):
    for base in ("/System/Library/Fonts/", "/Library/Fonts/"):
        p = Path(base) / "HelveticaNeue.ttc"
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size, index=1 if bold else 0)
            except OSError:
                pass
    print("    [warn] no TrueType face, falling back to bitmap")
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="metal-gauss")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=float, default=1.6)
    ap.add_argument("--out", default=str(ROOT / "assets" / "teaser.gif"))
    a = ap.parse_args()

    data = json.loads(SRC.read_text())
    fr = sorted((f for f in data["frames"] if f["impl"] == a.impl),
                key=lambda f: f["t"])
    if not fr:
        raise SystemExit(f"no frames for {a.impl}")
    for f in fr:
        f["img"] = Image.open(io.BytesIO(base64.b64decode(f["jpg"]))).convert("RGB")
    T = fr[-1]["t"]

    S = a.size
    f_big, f_small = _font(int(S * 0.075), True), _font(int(S * 0.045))
    tmp = Path(tempfile.mkdtemp())
    n_hold = int(a.hold * a.fps)
    for i in range(a.frames + n_hold):
        t = min(i, a.frames - 1) / (a.frames - 1) * T
        cur = fr[0]
        for f in fr:
            if f["t"] <= t:
                cur = f
        im = cur["img"].resize((S, S), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        # a dark strip so the readout stays legible over a white background
        d.rectangle([0, S - int(S * 0.16), S, S], fill=(12, 15, 19))
        pad = int(S * 0.035)
        d.text((pad, S - int(S * 0.135)), f"{t:5.1f}s", font=f_big, fill=(233, 230, 223))
        d.text((S - pad - d.textlength(f"{cur['psnr']:.2f} dB", font=f_big),
                S - int(S * 0.135)), f"{cur['psnr']:.2f} dB", font=f_big,
               fill=(217, 185, 78))
        im.save(tmp / f"f{i:04d}.png")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = tmp / "pal.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp / "f%04d.png"),
                    "-vf", "palettegen=max_colors=128:stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps),
                    "-i", str(tmp / "f%04d.png"), "-i", str(pal),
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", str(out)], check=True)
    print(f"  {a.impl}: {a.frames}+{n_hold} frames at {S}x{S} -> {out} "
          f"({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
