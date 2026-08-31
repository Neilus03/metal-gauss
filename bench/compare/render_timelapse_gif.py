"""Compose the side-by-side timelapse into a GIF for the README.

The interactive page (build_timelapse_page.py) is the better artefact, but
GitHub cannot embed it, so a repo landing page with no moving picture shows
nothing of the one claim that is hard to convey in a table. This renders the
same frames, the same "most recent checkpoint at time t" rule, and the same
DONE stamps into a loop that GitHub will play inline.

Kept deliberately small: the repo's history was slimmed to a few MB, so a
40 MB GIF would be a poor trade for a landing-page visual. Target is <5 MB.
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
# metal-gauss first: it is the subject, and readers scan left-to-right.
# Competitors follow in finish order. Equal wall-clock means every lane ends
# within ~60s of the others, so the order encodes nothing about the result.
LANES = [("metal-gauss", "metal-gauss"), ("brush", "Brush"),
         ("msplat", "msplat"), ("spirula", "spirula-studio")]
# Same palette as the HTML page: accent is lego's own DC colour.
COL = {"metal-gauss": (217, 185, 78), "msplat": (127, 190, 203),
       "brush": (188, 159, 184), "spirula": (143, 184, 148)}
BG, PANEL, INK, DIM = (17, 20, 25), (24, 28, 34), (233, 230, 223), (137, 144, 155)


def _font(size, bold=False):
    """Prefer a real face; fall back loudly rather than silently to a bitmap."""
    for name in (("HelveticaNeue.ttc",) if not bold else
                 ("HelveticaNeue.ttc",)):
        for base in ("/System/Library/Fonts/", "/Library/Fonts/"):
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size,
                                              index=1 if bold else 0)
                except OSError:
                    pass
    print(f"    [warn] no TrueType face found, falling back to bitmap at {size}px")
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "timelapse.gif"))
    # 4 lanes at 250px would be a 1048px-wide GIF; 200px keeps it near the
    # same total width as the 3-lane version and well under the size target.
    ap.add_argument("--panel", type=int, default=200)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds to hold the finished state before looping")
    a = ap.parse_args()

    data = json.loads(SRC.read_text())
    lanes = []
    for key, label in LANES:
        fr = sorted((f for f in data["frames"] if f["impl"] == key),
                    key=lambda f: f["t"])
        for f in fr:
            f["img"] = Image.open(io.BytesIO(base64.b64decode(f["jpg"]))).convert("RGB")
        lanes.append({"key": key, "label": label, "frames": fr})
    T_END = max(l["frames"][-1]["t"] for l in lanes)

    P = a.panel
    pad, head, foot = 12, 40, 44
    W = pad + (P + pad) * len(lanes)
    H = 62 + head + P + foot + pad
    f_lane, f_num, f_lab = _font(15, True), _font(17, True), _font(9)
    f_clock, f_stamp, f_note = _font(30, True), _font(15, True), _font(10)

    tmp = Path(tempfile.mkdtemp())
    n_hold = int(a.hold * a.fps)
    for i in range(a.frames + n_hold):
        t = min(i, a.frames - 1) / (a.frames - 1) * T_END
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        d.text((pad + 2, 12), "Splats against the clock", font=f_num, fill=INK)
        clock = f"{t:6.1f}s"
        d.text((W - pad - d.textlength(clock, font=f_clock), 8), clock,
               font=f_clock, fill=INK)
        d.text((pad + 2, 38), "lego  ·  held-out view  ·  metrics over 20 views @800px",
               font=f_note, fill=DIM)

        for j, ln in enumerate(lanes):
            x = pad + j * (P + pad)
            y = 62
            c = COL[ln["key"]]
            d.rectangle([x, y, x + P, y + head + P], fill=PANEL)
            d.rectangle([x, y, x + P, y + 3], fill=c)
            d.text((x + 9, y + 12), ln["label"], font=f_lane, fill=c)

            cur = None
            for f in ln["frames"]:
                if f["t"] <= t:
                    cur = f
            iy = y + head
            if cur is None:
                d.rectangle([x, iy, x + P, iy + P], fill=(11, 14, 18))
                msg = "awaiting first checkpoint"
                d.text((x + (P - d.textlength(msg, font=f_note)) / 2, iy + P / 2),
                       msg, font=f_note, fill=(90, 97, 107))
            else:
                im.paste(cur["img"].resize((P, P), Image.LANCZOS), (x, iy))
                # readout
                ry = iy + P + 6
                for k, (lab, val) in enumerate((
                        ("PSNR", f"{cur['psnr']:.2f}"),
                        ("SSIM", f"{cur['ssim']:.4f}"),
                        ("SPLATS", f"{cur['n']:,}"))):
                    cx = x + P * (k + .5) / 3
                    d.text((cx - d.textlength(lab, font=f_lab) / 2, ry), lab,
                           font=f_lab, fill=DIM)
                    d.text((cx - d.textlength(val, font=f_num) / 2, ry + 12), val,
                           font=f_num, fill=INK)
                # DONE stamp carries final quality: finishing first is not winning
                fin = ln["frames"][-1]
                if t >= fin["t"]:
                    txt, sub = "DONE", f"{fin['t']:.0f}s · {fin['psnr']:.2f} dB"
                    bw = max(d.textlength(txt, font=f_stamp),
                             d.textlength(sub, font=f_note)) + 16
                    bx, by = x + P - bw - 8, iy + P - 40
                    d.rectangle([bx, by, bx + bw, by + 34], fill=(15, 18, 22),
                                outline=c, width=2)
                    d.text((bx + (bw - d.textlength(txt, font=f_stamp)) / 2, by + 3),
                           txt, font=f_stamp, fill=c)
                    d.text((bx + (bw - d.textlength(sub, font=f_note)) / 2, by + 20),
                           sub, font=f_note, fill=(185, 191, 200))
        im.save(tmp / f"f{i:04d}.png")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = tmp / "pal.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp / "f%04d.png"),
                    "-vf", "palettegen=max_colors=128:stats_mode=diff", str(pal)],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps),
                    "-i", str(tmp / "f%04d.png"), "-i", str(pal),
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", str(out)],
                   check=True)
    mp4 = out.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps),
                    "-i", str(tmp / "f%04d.png"), "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "20", str(mp4)],
                   check=True)
    print(f"  {a.frames}+{n_hold} frames at {W}x{H}")
    print(f"    {out}  ({out.stat().st_size/1048576:.2f} MB)")
    print(f"    {mp4}  ({mp4.stat().st_size/1048576:.2f} MB)")


if __name__ == "__main__":
    main()
