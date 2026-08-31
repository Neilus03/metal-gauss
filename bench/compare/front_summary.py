"""Per-metric Pareto ownership across every scene that has been scored.

The front was published on PSNR and on lego alone. Both qualifiers turned out
to matter: PSNR inverted the ranking against msplat on lego, and the scenes
disagree with each other. This prints ownership per scene per metric so the
claim can be stated in exactly the form the data supports.

Reads bench/results/pareto_*_metrics.json. Pure analysis, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = (("psnr", True, "PSNR"), ("ssim", True, "SSIM"), ("lpips", False, "LPIPS"))


def front(rows, key, higher):
    pts = sorted([(r["wall_s"] / 60, r[key], r["impl"]) for r in rows])
    out, best = [], None
    for w, v, i in pts:
        if best is None or (v > best if higher else v < best):
            best = v
            out.append((w, v, i))
    return out


def dominated(rows, victim):
    """Points of `victim` beaten by metal-gauss on ALL THREE metrics at <= wall."""
    mg = [(r["wall_s"] / 60, r["psnr"], r["ssim"], r["lpips"])
          for r in rows if r["impl"] == "metal-gauss"]
    vic = [(r["wall_s"] / 60, r["psnr"], r["ssim"], r["lpips"])
           for r in rows if r["impl"] == victim]
    n = sum(1 for w, p, s, l in vic
            if any(wm <= w and pm >= p and sm >= s and lm <= l for wm, pm, sm, lm in mg))
    return n, len(vic)


def main():
    files = sorted((ROOT / "bench" / "results").glob("pareto_*_metrics.json"))
    if not files:
        raise SystemExit("no *_metrics.json found")
    scenes = {}
    for f in files:
        d = json.loads(f.read_text())
        rows = [r for r in d["rows"] if r.get("ssim") is not None]
        if not rows:
            continue
        name = (d.get("config") or {}).get("scene") or f.stem
        scenes[name] = rows

    print(f"{'scene':<10} {'metric':<7} {'front':>6}   ownership")
    for sc, rows in scenes.items():
        for key, higher, label in METRICS:
            f = front(rows, key, higher)
            own = {}
            for _, _, i in f:
                own[i] = own.get(i, 0) + 1
            share = "  ".join(f"{k} {v}" for k, v in sorted(own.items()))
            print(f"{sc:<10} {label:<7} {len(f):>6}   {share}")
        print()

    print("Points beaten by metal-gauss on ALL THREE metrics simultaneously:")
    for sc, rows in scenes.items():
        bits = []
        for victim in ("brush", "msplat"):
            n, tot = dominated(rows, victim)
            if tot:
                bits.append(f"{victim} {n}/{tot}")
        print(f"  {sc:<10} {'   '.join(bits)}")

    print("\nWinner at each wall-clock budget:")
    for sc, rows in scenes.items():
        print(f"  --- {sc} ---")
        for t in (0.5, 1.0, 2.0, 4.0, 14.0):
            cells = []
            for key, higher, label in METRICS:
                c = [(r[key], r["impl"]) for r in rows if r["wall_s"] / 60 <= t]
                if not c:
                    cells.append(f"{label}: --"); continue
                v, i = (max if higher else min)(c)
                cells.append(f"{label}: {i}")
            print(f"    <= {t:>5.1f} min   " + "   ".join(f"{c:<24}" for c in cells))


if __name__ == "__main__":
    main()
