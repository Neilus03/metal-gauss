"""8-scene mean Pareto front, PSNR vs wall-clock, rendered straight to SVG.

No matplotlib: the figure is a few dozen points, an SVG stays sharp in a README
at any zoom, and it diffs in git as text rather than as an opaque binary.

Two choices worth stating:

**Log x axis.** Budgets span 0.09 to 28 minutes, a 300x range. On a linear axis
everything below a minute -- which is where msplat lives and where the fast end
of the comparison is decided -- collapses into the y axis.

**Direct lines, not a staircase.** A step function would be right if these six
iteration counts were the only runnable configs, but iteration count is a
continuous knob -- an intermediate budget really does buy intermediate quality --
so joining points is both more faithful and more legible.

**The line is best-achievable-by-budget; every measurement is still plotted.**
Filled dots sit on the line, hollow ones are measured but dominated by a cheaper
run of the same implementation. Joining raw points in time order instead produced
spikes that looked like plotting errors but were real: msplat-stock's 2000-iter
run is both faster AND 6 dB better than its 1000-iter run, because its ADC prunes
and its resolution schedule steps. Hollow markers keep that visible without
letting it whip the line around.

The palette matches the timelapse page so a reader moving between them keeps the
same colour for the same implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]
# msplat's two variants are SEPARATE series. Merging them by taking the better
# PSNR per rung (which the README's budget table does, correctly, for "best
# achievable") produces a line that no run ever followed: the variants have very
# different wall-clock, so the merged sequence zigzags 19.9 -> 13.5 -> 21.2 in
# time order. A table of best-achievable may merge them; a trajectory may not.
SERIES = [
    ("metal-gauss", "metal-gauss", "#B8860B", "#D9B94E", False),
    ("spirula-studio", "spirula", "#3F6B4A", "#8FB894", False),
    ("Brush", "brush", "#6A4F6E", "#BC9FB8", False),
    ("msplat (stock)", "msplat-stock", "#26697A", "#7FBECB", False),
    ("msplat (scaled)", "msplat-scaled", "#26697A", "#7FBECB", True),
]
W, H = 820, 470
L, R, T, B = 62, 150, 30, 52


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "bench/results/sweep8"))
    ap.add_argument("--scenes", nargs="*", default=SCENES,
                    help="subset to average over; one scene renders that "
                         "scene's own front")
    ap.add_argument("--out", default=str(ROOT / "bench/results/pareto_8scene.svg"))
    a = ap.parse_args()
    root = Path(a.sweep)
    scenes = a.scenes

    def rows(scene, impl):
        f = root / f"{scene}_{impl}.json"
        if not f.exists():
            return []
        return [r for r in json.loads(f.read_text())["rows"]
                if r.get("ok") and r.get("psnr") is not None]

    series = []
    for label, impl, c_light, c_dark, dashed in SERIES:
        pts = []
        for n in (500, 1000, 2000, 4000, 7000, 15000):
            vals = [(r["wall_s"], r["psnr"]) for sc in scenes
                    for r in rows(sc, impl) if r["iters"] == n]
            if len(vals) < len(scenes):
                continue          # never average a partial set of scenes
            q = [v[1] for v in vals]
            mean = sum(q) / len(q)
            # Standard error of the 8-scene mean. NOT run-to-run noise, which is
            # far smaller (0.05 dB for us, 0.08 for spirula) -- this is how much
            # the mean would move on a different sample of scenes. Whiskers
            # overlap between implementations; that does not mean the difference
            # is uncertain, because the comparison is PAIRED (same scenes for
            # everyone) and the paired margin is much tighter. See the margin
            # table in the README.
            var = sum((x - mean) ** 2 for x in q) / (len(q) - 1) if len(q) > 1 else 0.0
            sem = (var / len(q)) ** 0.5
            pts.append((sum(v[0] for v in vals) / len(vals), mean, sem))
        series.append((label, c_light, c_dark, sorted(pts), dashed))

    allpts = [(p[0], p[1]) for _, _, _, ps, _ in series for p in ps]
    x0, x1 = min(p[0] for p in allpts) * 0.8, max(p[0] for p in allpts) * 1.15
    y0, y1 = math.floor(min(p[1] for p in allpts) - 1), math.ceil(max(p[1] for p in allpts) + 1)
    def X(w): return L + (math.log10(w) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * (W - L - R)
    def Y(q): return T + (1 - (q - y0) / (y1 - y0)) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="system-ui,-apple-system,sans-serif">',
         # GitHub honours prefers-color-scheme inside an inline SVG <style>, so
         # the figure follows the reader's theme instead of forcing a white card.
         '<style>',
         '  .bg{fill:#ffffff} .ax{stroke:#c9ccd2} .gr{stroke:#e8eaee}',
         '  .tx{fill:#3c4149} .ti{fill:#141920}',
         '  @media (prefers-color-scheme: dark){',
         '    .bg{fill:#0f1216} .ax{stroke:#39404a} .gr{stroke:#20252c}',
         '    .tx{fill:#98a0aa} .ti{fill:#e9e6df}',
         '    .s0{stroke:#D9B94E} .f0{fill:#D9B94E} .s1{stroke:#8FB894} .f1{fill:#8FB894}',
         '    .s2{stroke:#BC9FB8} .f2{fill:#BC9FB8} .s3{stroke:#7FBECB} .f3{fill:#7FBECB}',
         '    .s4{stroke:#7FBECB} .f4{fill:#7FBECB}',
         '  }']
    for i, (_, cl, _, _, _) in enumerate(series):
        # Stroke and fill are separate classes on purpose: a single class
        # setting both would override fill="none" on the staircase paths,
        # because CSS wins over presentation attributes.
        o.append(f'  .s{i}{{stroke:{cl};fill:none}} .f{i}{{fill:{cl};stroke:none}}')
    o.append('</style>')
    o.append(f'<rect class="bg" width="{W}" height="{H}"/>')

    for q in range(y0, y1 + 1, 2):
        o.append(f'<line class="gr" x1="{L}" y1="{Y(q):.1f}" x2="{W-R}" y2="{Y(q):.1f}"/>')
        o.append(f'<text class="tx" x="{L-10}" y="{Y(q)+4:.1f}" font-size="11" '
                 f'text-anchor="end">{q}</text>')
    for w, lab in ((6, "0.1 min"), (30, "0.5"), (60, "1"), (180, "3"),
                   (600, "10"), (1800, "30")):
        if not (x0 <= w <= x1):
            continue
        o.append(f'<line class="gr" x1="{X(w):.1f}" y1="{T}" x2="{X(w):.1f}" y2="{H-B}"/>')
        o.append(f'<text class="tx" x="{X(w):.1f}" y="{H-B+18}" font-size="11" '
                 f'text-anchor="middle">{lab}</text>')
    o.append(f'<line class="ax" x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}"/>')
    o.append(f'<line class="ax" x1="{L}" y1="{T}" x2="{L}" y2="{H-B}"/>')
    o.append(f'<text class="tx" x="{(L+W-R)/2:.0f}" y="{H-12}" font-size="12" '
             f'text-anchor="middle">wall-clock (minutes, log scale)</text>')
    ylab = f'{len(scenes)}-scene mean' if len(scenes) > 1 else scenes[0]
    o.append(f'<text class="tx" transform="translate(16,{(T+H-B)/2:.0f}) rotate(-90)" '
             f'font-size="12" text-anchor="middle">PSNR (dB), {ylab}</text>')

    for i, (label, _, _, pts, dashed) in enumerate(series):
        # The LINE traces best-achievable-by-budget; every measured point is
        # still drawn. Joining the raw points in time order instead produced
        # spikes that read as errors but were real: msplat-stock's 2000-iter run
        # is both faster AND 6 dB better than its 1000-iter run, because its ADC
        # prunes and its resolution schedule steps. Those points now sit visibly
        # below the line rather than dragging it up and down.
        env, best = [], -1e9
        for w, q, _sem in pts:
            if q > best:
                env.append((w, q))
                best = q
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(w):.1f},{Y(q):.1f}"
                     for k, (w, q) in enumerate(env))
        dash = ' stroke-dasharray="5 4"' if dashed else ''
        # Faint line through EVERY measured point, in wall-clock order, under the
        # bold envelope. Without it the dominated points read as orphans floating
        # near the chart rather than as part of a run: a hollow dot below the line
        # is that implementation actually getting worse with more time, and it
        # needs something to belong to for that to be legible.
        if len(pts) > 1:
            dt = " ".join(f"{'M' if k == 0 else 'L'}{X(w):.1f},{Y(q):.1f}"
                          for k, (w, q, _s) in enumerate(pts))
            o.append(f'<path class="s{i}" d="{dt}" stroke-width="1" '
                     f'stroke-opacity="0.28"{dash}/>')
        o.append(f'<path class="s{i}" d="{d}" stroke-width="{2.4 if i == 0 else 1.8}" '
                 f'stroke-linejoin="round" stroke-linecap="round"{dash} '
                 f'stroke-opacity="{0.95 if i == 0 else 0.7}"/>')
        # SEM whiskers, drawn under the markers.
        for w, q, sem in pts:
            if sem > 0:
                o.append(f'<line class="s{i}" x1="{X(w):.1f}" y1="{Y(q-sem):.1f}" '
                         f'x2="{X(w):.1f}" y2="{Y(q+sem):.1f}" stroke-width="1.2" '
                         f'stroke-opacity="0.5"/>')
        on = {p for p in env}
        for w, q, _sem in pts:
            # Dominated points are drawn hollow: measured, but not the best you
            # can do at that budget.
            if (w, q) in on:
                o.append(f'<circle class="f{i}" cx="{X(w):.1f}" cy="{Y(q):.1f}" '
                         f'r="{4 if i == 0 else 3}"/>')
            else:
                o.append(f'<circle class="s{i}" cx="{X(w):.1f}" cy="{Y(q):.1f}" '
                         f'r="2.6" stroke-width="1.3" fill="none" '
                         f'stroke-opacity="0.65"/>')
        ly = T + 14 + i * 20
        o.append(f'<line class="s{i}" x1="{W-R+8}" y1="{ly-4}" x2="{W-R+26}" '
                 f'y2="{ly-4}" stroke-width="3"{dash}/>')
        o.append(f'<text class="ti" x="{W-R+32}" y="{ly}" font-size="12" '
                 f'font-weight="{700 if i == 0 else 400}">{label}</text>')
    ky = T + 14 + len(series) * 20 + 12
    o.append(f'<circle class="f0" cx="{W-R+16}" cy="{ky-4}" r="3.5" '
             f'fill-opacity="0.75"/>')
    o.append(f'<text class="tx" x="{W-R+32}" y="{ky}" font-size="10.5">best at that budget</text>')
    o.append(f'<circle class="s0" cx="{W-R+16}" cy="{ky+15:.0f}" r="2.6" fill="none" '
             f'stroke-width="1.3" stroke-opacity="0.65"/>')
    o.append(f'<text class="tx" x="{W-R+32}" y="{ky+19:.0f}" font-size="10.5">beaten by a</text>')
    o.append(f'<text class="tx" x="{W-R+32}" y="{ky+31:.0f}" font-size="10.5">cheaper run</text>')
    o.append('</svg>')
    Path(a.out).write_text("\n".join(o))
    print(f"  {sum(len(p) for _,_,_,p,_ in series)} points -> {a.out} "
          f"({Path(a.out).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
