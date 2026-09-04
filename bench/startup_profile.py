"""How much of a short run is startup, and what is cached between runs.

msplat wins every wall-clock budget under ~0.3 min on lego, and a 100-iteration
run costs 0.22 min of which the training is a few seconds. That makes startup,
not throughput, the binding constraint at the fast end -- so it is worth
knowing what startup is made of.

Startup is measured as harness_wall_s - train_only_wall_s: the trainer's own
timer begins after scene loading and kernel setup, so the difference is
everything before the first step.

Three candidates, separated here:
  * the C++/ObjC++ extension build (torch cpp_extension, cached on disk as a
    .so, so it should be paid once ever);
  * Metal library loading: installed packages use a matching precompiled
    ``.metallib`` when present; ``METAL_GAUSS_FORCE_SOURCE=1`` forces the
    ``newLibraryWithSource`` fallback for comparison;
  * filesystem page cache for the scene images, warm after the first read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.runner import run  # noqa: E402
from bench.provenance import env as _env  # noqa: E402


def ext_cache_dir() -> Path | None:
    try:
        from torch.utils.cpp_extension import _get_build_directory
        return Path(_get_build_directory("metal_gauss_metal", False))
    except Exception:
        return None


def one(scene: Path, steps: int) -> dict:
    rep = run({"blender": str(scene), "steps": steps,
               "max_resolution": 800, "eval_every": steps * 10})
    total = rep["harness_wall_s"]
    train = rep["metrics"]["wall_s"]
    return {"harness_s": round(total, 2), "train_s": round(train, 2),
            "startup_s": round(total - train, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--steps", type=int, default=50,
                    help="tiny, so startup dominates and is easy to read")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--cold-ext", action="store_true",
                    help="move the built .so aside first, forcing a rebuild, "
                         "to price the extension build separately")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "startup_profile.json"))
    a = ap.parse_args()

    scene, rows = Path(a.scene), []
    print(f"scene {scene.name}, {a.steps} steps, {a.repeats} consecutive runs\n")

    if a.cold_ext:
        d = ext_cache_dir()
        if d and d.exists():
            stash = d.with_name(d.name + ".stashed")
            shutil.rmtree(stash, ignore_errors=True)
            shutil.move(str(d), str(stash))
            print(f"  extension cache moved aside -> {stash.name}\n")
        else:
            print("  no extension cache found; --cold-ext is a no-op\n")

    for i in range(a.repeats):
        r = one(scene, a.steps)
        r["run"] = i
        r["ext_cold"] = bool(a.cold_ext and i == 0)
        rows.append(r)
        tag = "  <- extension rebuilt" if r["ext_cold"] else ""
        print(f"  run {i}  startup {r['startup_s']:6.2f} s   "
              f"train {r['train_s']:6.2f} s   total {r['harness_s']:6.2f} s{tag}",
              flush=True)

    warm = [r["startup_s"] for r in rows if not r["ext_cold"]]
    if warm:
        print(f"\n  warm startup: min {min(warm):.2f} s, max {max(warm):.2f} s, "
              f"spread {max(warm) - min(warm):.2f} s over {len(warm)} runs")
    if a.cold_ext and rows:
        # The rebuild does NOT show up in startup_s. torch's cpp_extension
        # loads lazily on first kernel use, which happens after the trainer
        # has started its own timer, so the cost lands in train_s. Reading it
        # from startup_s -- as the first version of this line did -- reports
        # roughly zero and hides a 12 s effect.
        warm_train = [r["train_s"] for r in rows if not r["ext_cold"]]
        if warm_train:
            print(f"  extension build cost: "
                  f"{rows[0]['train_s'] - min(warm_train):.2f} s, and it appears "
                  f"in train_s not startup_s (lazy load, after the timer starts). "
                  f"Paid once; cached on disk afterwards.")

    Path(a.out).write_text(json.dumps(
        {"schema": 1, "env": _env(),
         "config": {"scene": scene.name, "steps": a.steps,
                    "repeats": a.repeats, "cold_ext": a.cold_ext},
         "rows": rows}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
