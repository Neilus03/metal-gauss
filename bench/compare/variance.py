"""Repeat one configuration N times to measure an implementation's noise floor.

This exists because a single run was used to claim a competitor beat us, and
separately to claim one of its variants beat another by +4.50 dB. Neither
survived a second run. metal-gauss reproduces to 0.22 dB and Brush to 0.74, but
msplat spreads up to 3.35 dB with no seed flag to control it, so "A beats B by
1 dB" is not a statement any single pair of runs can support.

spirula is the urgent case: it wins mic outright and ties lego, and it had zero
repeats when those claims were first written down.

Every repeat gets a distinct --tag so runs cannot overwrite one another; that
collision has already destroyed data three times in this repo.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runner import require_gpu_exclusive  # noqa: E402

OUT = ROOT / "bench" / "results" / "variance"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True,
                    choices=["metal-gauss", "msplat-stock", "msplat-scaled",
                             "brush", "spirula"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=800)
    a = ap.parse_args()

    require_gpu_exclusive()
    OUT.mkdir(parents=True, exist_ok=True)
    blender = ROOT / "data" / "nerf_synthetic" / a.scene
    ns = f"/tmp/cmp_data/{a.scene}_ns"
    rows = []
    for i in range(a.repeats):
        tag = f"_v{i}"
        out = OUT / f"{a.scene}_{a.impl}_{a.iters}{tag}.json"
        if out.exists():
            rows.append(json.loads(out.read_text()))
            print(f"  [{i+1}/{a.repeats}] cached", flush=True)
            continue
        if a.impl == "spirula":
            cmd = [sys.executable, ROOT / "bench/compare/run_spirula.py",
                   "--scene", blender, "--data", ns, "--iters", a.iters,
                   "--resolution", a.resolution, "--tag", tag, "--out", out]
        else:
            cmd = [sys.executable, ROOT / "bench/compare/pareto.py",
                   "--scene", blender, "--scene-ns", ns, "--iters", a.iters,
                   "--resolution", a.resolution, "--tag", tag, "--out", out,
                   "--impls", a.impl.split("-")[0] if a.impl.startswith("msplat")
                   else a.impl]
            if a.impl == "msplat-stock":
                cmd.append("--msplat-stock")
        t0 = time.perf_counter()
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                           capture_output=True, text=True)
        el = time.perf_counter() - t0
        if not out.exists():
            tail = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1]
            print(f"  [{i+1}/{a.repeats}] FAILED {tail[:100]}", flush=True)
            continue
        d = json.loads(out.read_text())
        rows.append(d)
        r = [x for x in d["rows"] if x.get("ok")]
        if r:
            print(f"  [{i+1}/{a.repeats}] {r[-1]['psnr']:6.3f} dB  "
                  f"{r[-1]['wall_s']/60:5.2f} min  ({el/60:.1f} min total)", flush=True)

    vals = [x["psnr"] for d in rows for x in d["rows"]
            if x.get("ok") and x.get("psnr")]
    if len(vals) < 2:
        print("  too few successful runs to summarise")
        return
    print(f"\n  {a.impl} / {a.scene} / {a.iters} it  (n={len(vals)})")
    print(f"    mean  {statistics.mean(vals):.3f} dB")
    print(f"    stdev {statistics.stdev(vals):.3f} dB")
    print(f"    range {min(vals):.3f} .. {max(vals):.3f}  "
          f"(spread {max(vals)-min(vals):.3f} dB)")


if __name__ == "__main__":
    main()
