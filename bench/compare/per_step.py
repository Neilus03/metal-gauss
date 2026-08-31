"""Is msplat's rasteriser faster than ours, or is it just doing less work?

Every comparison in this repo has confounded three things: splat count,
resolution schedule and densification. At 7k iterations msplat takes 60.6s to
our 188s while carrying 64k splats to our 100k and finishing 6.5 dB worse, so
"they are faster" has never actually been established either way.

This pins all three and measures per-step cost directly.

    splat count   ours --budget 100000 --no-grow;  theirs --warmup-length > iters
    resolution    --num-downscales 0 on both
    SH degree     ours --sh-warmup 0;  theirs --sh-degree-interval 1
    init, scene   the same sparse.ply, lego

Two flag details that would silently corrupt this, both verified in the source:

  * train.py:333 is `active = min(start_active, budget) if grow else budget`,
    so under --no-grow the count is `budget` and --start-active does NOTHING.
    It is not passed, so nothing implies otherwise.
  * train.py:360 is `sh_deg = min(3, step // sh_warmup)`, so per-step cost
    RISES over the first 1000 steps as the SH degree ramps. Left at the default
    the linear fit below would be fitting that ramp, not the rasteriser. Both
    sides have it disabled.

Per-step cost is recovered as the SLOPE of wall against iteration count, not by
subtracting a startup estimate: it needs no instrumentation inside either
trainer and no trust in either one's progress output. The intercept is startup,
which measures msplat's figure -- previously asserted at "~6 s" and never
measured -- and independently cross-checks our own 8.4 s.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runner import RunDiverged, RunFailed, run  # noqa: E402
from bench.provenance import env as _env  # noqa: E402

MSPLAT = "/tmp/cmp_msplat/bin/msplat-train"


def fit(ns, walls):
    """Least squares wall = a + b*n. Returns (intercept_s, slope_ms, r2)."""
    k = len(ns)
    mx = sum(ns) / k
    my = sum(walls) / k
    sxx = sum((x - mx) ** 2 for x in ns)
    sxy = sum((x - mx) * (y - my) for x, y in zip(ns, walls))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in walls)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(ns, walls))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return a, b * 1000.0, r2


def ply_count(path: Path) -> int | None:
    """Splat count from the ply header, to prove densification stayed off."""
    try:
        with open(path, "rb") as f:
            for _ in range(64):
                line = f.readline().decode("ascii", "ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--scene-ns", default="/tmp/cmp_data/lego_ns")
    ap.add_argument("--iters", nargs="*", type=int, default=[100, 300, 600, 1000])
    ap.add_argument("--budget", type=int, default=100_000)
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--no-mcmc", action="store_true",
                    help="also disable MCMC relocation and SGLD noise on our "
                         "side. Without this the comparison is our step WITH "
                         "densification against theirs WITHOUT any, since "
                         "--warmup-length beyond the run stops theirs "
                         "entirely. Needed for a rasteriser-only ratio.")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "per_step.json"))
    a = ap.parse_args()

    scene = Path(a.scene)
    outdir = Path("/tmp/cmp_out"); outdir.mkdir(exist_ok=True)
    rows = []

    print(f"matched: {a.budget:,} splats, full resolution, SH ramp off, lego\n")

    # ---- metal-gauss -------------------------------------------------------
    mg_ns, mg_walls = [], []
    for n in a.iters:
        spec = {"blender": str(scene), "steps": n, "budget": a.budget,
                "grow": False, "num_downscales": 0, "sh_warmup": 0,
                "max_resolution": a.resolution, "eval_every": n * 10}
        if a.no_mcmc:
            spec["relocate_until_frac"] = 0.0
            spec["noise_weight"] = 0.0
        try:
            rep = run(spec)
        except (RunFailed, RunDiverged) as e:
            print(f"  metal-gauss {n:>5}  FAILED {str(e)[:100]}")
            rows.append({"impl": "metal-gauss", "iters": n, "ok": False,
                         "error": str(e)[:400]})
            continue
        w = rep["harness_wall_s"]
        res = rep["resolved"]
        assert res["budget"] == a.budget and res["grow"] is False, \
            f"count not pinned: budget={res['budget']} grow={res['grow']}"
        mg_ns.append(n); mg_walls.append(w)
        print(f"  metal-gauss {n:>5}  {w:7.2f}s  ({rep['metrics']['n_splats']:,} splats)",
              flush=True)
        rows.append({"impl": "metal-gauss", "iters": n, "ok": True, "wall_s": w,
                     "n_splats": rep["metrics"]["n_splats"],
                     "resolved": res, "env": rep["env"]})

    # ---- msplat ------------------------------------------------------------
    ms_ns, ms_walls = [], []
    for n in a.iters:
        ply = outdir / f"perstep_msplat_{n}.ply"
        if ply.exists():
            ply.unlink()
        cmd = [MSPLAT, "--input", a.scene_ns, "--num-iters", str(n),
               "--keep-crs", "--output", str(ply),
               "--num-downscales", "0",
               "--sh-degree-interval", "1",
               # densification must never start, so the count stays at the
               # initialisation and matches ours
               "--warmup-length", str(n + 10)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
        w = time.perf_counter() - t0
        cnt = ply_count(ply)
        if p.returncode != 0 or cnt is None:
            tail = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1]
            print(f"  msplat      {n:>5}  FAILED  {tail[:100]}")
            rows.append({"impl": "msplat", "iters": n, "ok": False, "tail": tail[:400]})
            continue
        # a moved count means densification ran and the point is void
        drift = abs(cnt - a.budget) / a.budget
        ok = drift < 0.05
        ms_ns.append(n) if ok else None
        ms_walls.append(w) if ok else None
        flag = "" if ok else "   <- COUNT DRIFTED, EXCLUDED"
        print(f"  msplat      {n:>5}  {w:7.2f}s  ({cnt:,} splats){flag}", flush=True)
        rows.append({"impl": "msplat", "iters": n, "ok": ok, "wall_s": w,
                     "n_splats": cnt, "count_drift": round(drift, 4),
                     "cmd": " ".join(cmd)})

    out = {"schema": 1, "env": _env(),
           "config": {"budget": a.budget, "resolution": a.resolution,
                      "iters": a.iters, "scene": scene.name},
           "rows": rows, "fits": {}}

    print()
    for name, ns, ws in (("metal-gauss", mg_ns, mg_walls), ("msplat", ms_ns, ms_walls)):
        if len(ns) < 2:
            print(f"  {name}: too few points to fit"); continue
        a0, b, r2 = fit(ns, ws)
        out["fits"][name] = {"startup_s": round(a0, 2), "ms_per_step": round(b, 3),
                             "r2": round(r2, 5), "points": len(ns)}
        warn = "   <- POOR FIT, slope not meaningful" if r2 < 0.95 else ""
        print(f"  {name:<12} startup {a0:6.2f}s   {b:7.3f} ms/step   R2 {r2:.4f}{warn}")

    f = out["fits"]
    if "metal-gauss" in f and "msplat" in f:
        ratio = f["metal-gauss"]["ms_per_step"] / f["msplat"]["ms_per_step"]
        print(f"\n  at {a.budget:,} splats, full resolution, no densification:")
        print(f"    metal-gauss is {ratio:.2f}x msplat's per-step cost")
        print(f"    startup: ours {f['metal-gauss']['startup_s']:.1f}s vs "
              f"theirs {f['msplat']['startup_s']:.1f}s")
        out["ratio_per_step"] = round(ratio, 3)

    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
