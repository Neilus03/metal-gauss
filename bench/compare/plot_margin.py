"""Forest plot of the paired per-scene margin, metal-gauss minus each competitor.

A table of nine confidence intervals is a table of nine confidence intervals. The
question a reader has -- does this margin clear zero? -- is a spatial one, and a
forest plot answers it by putting a line at zero and letting every interval sit
against it.

Paired is the whole point. Each implementation ran the SAME eight scenes, so the
statistic is the spread of the per-scene DIFFERENCE. The spread of either mean is
dominated by scene difficulty (SD 2.4-5.4 dB) and would imply no difference where
there is a large one.

95% CI is Student-t with 7 degrees of freedom (t = 2.365).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]
COMP = [("spirula", "spirula-studio", "#3F6B4A", "#8FB894"),
        ("brush", "Brush", "#6A4F6E", "#BC9FB8"),
        ("msplat", "msplat", "#26697A", "#7FBECB")]
RUNGS = [500, 7000, 15000]
W, H = 880, 352
# R is wide because the right-hand annotation carries the words
# "crosses 0", which is the single most important thing on this chart.
L, R, T, B = 168, 168, 52, 46


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "bench/results/sweep8"))
    ap.add_argument("--out", default=str(ROOT / "bench/results/margin_forest.svg"))
    a = ap.parse_args()
    root = Path(a.sweep)

    def best(scene, key, iters):
        keys = ["msplat-stock", "msplat-scaled"] if key == "msplat" else [key]
        vals = []
        for k in keys:
            f = root / f"{scene}_{k}.json"
            if not f.exists():
                continue
            vals += [r["psnr"] for r in json.loads(f.read_text())["rows"]
                     if r.get("ok") and r.get("psnr") is not None and r["iters"] == iters]
        return max(vals) if vals else None

    rows = []
    for key, label, cl, cd in COMP:
        for n in RUNGS:
            d = [best(s, "metal-gauss", n) - best(s, key, n) for s in SCENES
                 if best(s, "metal-gauss", n) is not None and best(s, key, n) is not None]
            if len(d) != len(SCENES):
                continue
            m = sum(d) / len(d)
            var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
            sem = (var / len(d)) ** 0.5
            rows.append({"label": label, "n": n, "cl": cl, "cd": cd, "m": m,
                         "lo": m - 2.365 * sem, "hi": m + 2.365 * sem,
                         "won": sum(1 for x in d if x > 0)})

    lo = min(r["lo"] for r in rows) - 1
    hi = max(r["hi"] for r in rows) + 1
    lo = min(lo, -1.5)
    def X(v): return L + (v - lo) / (hi - lo) * (W - L - R)
    step = (H - T - B) / len(rows)
    def Y(i): return T + step * (i + 0.5)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
         f'height="{H}" font-family="system-ui,-apple-system,sans-serif">',
         '<style>',
         '  .bg{fill:#ffffff} .gr{stroke:#e8eaee} .zero{stroke:#8b929c}',
         '  .tx{fill:#3c4149} .ti{fill:#141920} .warn{fill:#8A5A00}',
         '  @media (prefers-color-scheme: dark){',
         '    .bg{fill:#0f1216} .gr{stroke:#20252c} .zero{stroke:#6b727c}',
         '    .tx{fill:#98a0aa} .ti{fill:#e9e6df} .warn{fill:#D9B94E}',
         '    .c0{stroke:#8FB894;fill:#8FB894} .c1{stroke:#BC9FB8;fill:#BC9FB8}',
         '    .c2{stroke:#7FBECB;fill:#7FBECB}',
         '  }']
    for i, (_, _, cl, _) in enumerate(COMP):
        o.append(f'  .c{i}{{stroke:{cl};fill:{cl}}}')
    o.append('</style>')
    o.append(f'<rect class="bg" width="{W}" height="{H}"/>')
    o.append(f'<text class="ti" x="14" y="22" font-size="13" font-weight="700">'
             f'metal-gauss minus competitor, paired per scene (n=8)</text>')

    for v in range(int(lo) + 1, int(hi) + 1, 2):
        o.append(f'<line class="gr" x1="{X(v):.1f}" y1="{T-6}" x2="{X(v):.1f}" y2="{H-B+4}"/>')
        o.append(f'<text class="tx" x="{X(v):.1f}" y="{H-B+20}" font-size="10.5" '
                 f'text-anchor="middle">{v:+d}</text>')
    # the line that matters
    o.append(f'<line class="zero" x1="{X(0):.1f}" y1="{T-10}" x2="{X(0):.1f}" '
             f'y2="{H-B+4}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    o.append(f'<text class="tx" x="{X(0):.1f}" y="{T-14}" font-size="10.5" '
             f'text-anchor="middle">no difference</text>')
    o.append(f'<text class="tx" x="{(L+W-R)/2:.0f}" y="{H-10}" font-size="11.5" '
             f'text-anchor="middle">PSNR advantage (dB), 95% CI</text>')

    ci_idx = {lbl: i for i, (_, lbl, _, _) in enumerate(COMP)}
    for i, r in enumerate(rows):
        y, c = Y(i), ci_idx[r["label"]]
        crosses = r["lo"] <= 0 <= r["hi"]
        o.append(f'<text class="ti" x="{L-12}" y="{y+4:.1f}" font-size="11.5" '
                 f'text-anchor="end">{r["label"]} @ {r["n"]//1000 if r["n"]>=1000 else r["n"]}'
                 f'{"k" if r["n"]>=1000 else ""}</text>')
        o.append(f'<line class="c{c}" x1="{X(r["lo"]):.1f}" y1="{y:.1f}" '
                 f'x2="{X(r["hi"]):.1f}" y2="{y:.1f}" stroke-width="2.2" '
                 f'stroke-opacity="0.8" stroke-linecap="round"/>')
        for e in ("lo", "hi"):
            o.append(f'<line class="c{c}" x1="{X(r[e]):.1f}" y1="{y-4:.1f}" '
                     f'x2="{X(r[e]):.1f}" y2="{y+4:.1f}" stroke-width="1.6" '
                     f'stroke-opacity="0.8"/>')
        o.append(f'<circle class="c{c}" cx="{X(r["m"]):.1f}" cy="{y:.1f}" r="4.5"/>')
        cls = "warn" if crosses else "tx"
        note = f'{r["m"]:+.2f}  ({r["won"]}/8)' + ("  crosses 0" if crosses else "")
        o.append(f'<text class="{cls}" x="{W-R+8}" y="{y+4:.1f}" font-size="10.5">{note}</text>')
    o.append('</svg>')
    Path(a.out).write_text("\n".join(o))
    print(f"  {len(rows)} comparisons -> {a.out} ({Path(a.out).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
