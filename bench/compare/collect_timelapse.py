"""Checkpoints from all three trainers, indexed by WALL-CLOCK, for the
convergence comparison.

Every result in this repo is a number in a table. The claim those numbers make
-- that we reach a given quality soonest -- is far easier to see than to read,
so this collects the material for a side-by-side animation.

Two decisions worth stating.

**Wall-clock, not iteration.** Animating by step would show three methods
finishing together and hide the entire point. The x axis is elapsed seconds.

**Timing comes from file mtime, not from interpolating the step count.** None
of the three trainers timestamps its checkpoints, and per-step cost is not
constant -- msplat's ADC prunes 100k splats to 19k partway through, after which
its steps get markedly cheaper. Interpolating would place its frames wrongly
and either flatter or penalise it depending on the direction.

Runs are strictly sequential. The output is a wall-clock claim, so a contended
GPU would invalidate all of it.
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
from bench.provenance import env as _env  # noqa: E402
from bench.runner import require_gpu_exclusive  # noqa: E402

from bench.paths import brush_bin, msplat_bin, spirula_bin  # noqa: E402

# EQUAL WALL-CLOCK, not equal iterations. Pinning every lane to the same step
# count is exactly the matched-iteration comparison this repo exists to reject,
# and it is what made msplat "finish" at 66s while metal-gauss ran to 187s --
# reading as though msplat had won the race by stopping first.
#
# Each lane instead gets ~390 seconds and runs however many iterations fit,
# sized from the 8-scene sweep's measured lego timings:
#   metal-gauss 15000 -> 356s      brush        7000 -> 373s
#   msplat      15000 -> 279s      spirula      4000 -> 212s, 7000 -> 585s
# msplat and spirula are interpolated to land near 390s. Slight over/undershoot
# is harmless: the animation plots real elapsed time, so a lane that finishes at
# 370s or 410s simply stamps DONE there.
WALL_BUDGET_S = 390
STEPS = {"metal-gauss": 15000, "msplat-ladder": 19000, "brush": 7000,
         "spirula": 5500}


def collect(impl: str, scene: Path, scene_ns: str, steps: int, every: int,
            outdir: Path, res: int) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.ply"):
        f.unlink()

    if impl == "msplat-ladder":
        # msplat's --save-every OVERWRITES --output rather than writing
        # numbered files, so one run yields one checkpoint. Instead run a
        # ladder of separate runs at increasing iteration counts, each
        # contributing one point.
        #
        # This is fair rather than a workaround: metal-gauss's t0 is process
        # start, so its checkpoint times already include its startup, and each
        # ladder run includes its own. Every method's elapsed time therefore
        # means the same thing -- time from launch to reaching this state.
        # Rungs must cover the WALL-CLOCK span evenly, not the iteration span.
        # The original hardcoded list stopped at 6000 and jumped to `steps`;
        # at steps=19000 that left msplat's lane frozen from 40s to 379s, which
        # reads as a crashed run rather than a slow one. msplat's cost is
        # roughly quadratic in iterations here (6000 -> 40s, 19000 -> 379s), so
        # evenly spaced TIME needs rungs bunched toward the top end.
        low = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
        rungs = [r for r in low if r <= steps]
        if steps > 6000:
            n_hi = 8
            rungs += [int(round(steps * (t / n_hi) ** 0.51 / 100)) * 100
                      for t in range(1, n_hi + 1)]
        rungs = [r for r in rungs if r <= steps] + [steps]
        rows = []
        for n in sorted(set(r for r in rungs if r <= steps)):
            dst = outdir / f"ms_{n:06d}.ply"
            t1 = time.time()
            q = subprocess.run([msplat_bin(), "--input", scene_ns, "--num-iters", str(n),
                                "--keep-crs", "--output", str(dst)],
                               capture_output=True, text=True, cwd="/tmp")
            el = time.time() - t1
            if not dst.exists():
                tail = ((q.stderr or q.stdout).strip().splitlines() or ["?"])[-1]
                print(f"  msplat {n:>6}: FAILED {tail[:90]}", flush=True)
                continue
            rows.append({"impl": "msplat", "ply": str(dst), "t": round(el, 2),
                         "iters": n})
            print(f"  msplat {n:>6} iters  {el:6.1f}s", flush=True)
        if rows:
            print(f"  {'msplat':<12} {len(rows):>3} ladder points, "
                  f"last at {rows[-1]['t']:.1f}s", flush=True)
            rows.append({"impl": "msplat", "done_at": rows[-1]["t"]})
        return rows

    if impl == "spirula":
        # --save-only-latest-checkpoint defaults to 1: it DELETES older
        # checkpoints as training proceeds, which would yield a one-frame
        # timelapse. --disable-viewer is equally mandatory; without it the
        # process serves an HTTP viewer forever after training ends and the
        # measured wall-clock is the viewer, not the run.
        cmd = [spirula_bin(), "train", "synthetic", "--data", scene_ns,
               "--data-format", "nerfstudio", "--num-iterations", str(steps),
               "--sh-degree", "3", "--disable-viewer", "1",
               "--steps-per-save", str(every),
               "--save-only-latest-checkpoint", "0",
               "--output-dir-prefix", str(outdir), "--output-dir-name", "tl"]
        cwd = "/tmp"
    elif impl == "metal-gauss":
        cmd = [sys.executable, "-m", "metal_gauss.train", "--blender", str(scene),
               "--steps", str(steps), "--max-resolution", str(res),
               "--eval-every", str(steps * 10),
               "--export", str(outdir / "mg.ply"), "--export-every", str(every)]
        cwd = ROOT
    elif impl == "msplat":
        cmd = [msplat_bin(), "--input", scene_ns, "--num-iters", str(steps),
               "--keep-crs", "--output", str(outdir / "ms.ply"),
               "--save-every", str(every)]
        cwd = "/tmp"
    elif impl == "brush":
        cmd = [brush_bin(), str(scene), "--total-steps", str(steps),
               "--max-resolution", str(res), "--sh-degree", "3",
               "--export-every", str(every), "--export-path", str(outdir),
               "--export-name", "br_{iter}.ply"]
        cwd = str(outdir)
    else:
        raise SystemExit(f"unknown impl {impl}")

    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    total = time.time() - t0
    # spirula nests each checkpoint in its own step-NNNNNN.ckpt directory.
    plys = sorted(outdir.glob("**/*.ply"), key=lambda f: f.stat().st_mtime)
    if not plys:
        tail = ((p.stderr or p.stdout).strip().splitlines() or ["?"])[-1]
        print(f"  {impl}: NO CHECKPOINTS  {tail[:120]}", flush=True)
        return []
    rows = [{"impl": impl, "ply": str(f), "t": round(f.stat().st_mtime - t0, 2)}
            for f in plys]
    # a checkpoint cannot precede the run that produced it
    rows = [r for r in rows if r["t"] >= 0]
    print(f"  {impl:<12} {len(rows):>3} checkpoints over {total/60:.2f} min "
          f"(last at {rows[-1]['t']:.1f}s)", flush=True)
    return rows + [{"impl": impl, "done_at": round(total, 2)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--scene-ns", default="/tmp/cmp_data/lego_ns")
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = per-implementation counts sized for an equal "
                         "WALL-CLOCK budget (see STEPS); a value here forces "
                         "the old matched-iteration behaviour")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--res", type=int, default=800)
    ap.add_argument("--impls", nargs="*",
                    default=["metal-gauss", "msplat-ladder", "brush", "spirula"])
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "timelapse.json"))
    a = ap.parse_args()

    require_gpu_exclusive()          # every row here is a wall-clock claim
    scene = Path(a.scene)
    allrows = []
    used = {}
    mode = ("equal wall-clock, ~%ds/lane" % WALL_BUDGET_S) if not a.steps \
        else f"matched iterations ({a.steps})"
    print(f"{scene.name}, {mode}, checkpoint every {a.every}, "
          f"strictly sequential\n")
    for impl in a.impls:
        steps = a.steps or STEPS.get(impl)
        if steps is None:
            print(f"  {impl}: no step count configured, skipped", flush=True)
            continue
        used[impl] = steps
        # Checkpoint density is scaled per lane so every lane yields a similar
        # NUMBER of frames despite running different step counts -- otherwise
        # the lane with the most iterations animates smoothly and the others
        # stutter, which reads as a quality difference rather than a schedule one.
        every = max(1, round(steps / (STEPS["metal-gauss"] / a.every)))
        allrows += collect(impl, scene, a.scene_ns, steps, every,
                           Path(f"/tmp/timelapse/{impl}"), a.res)
        Path(a.out).write_text(json.dumps(
            {"schema": 1, "env": _env(),
             "config": {"scene": scene.name, "steps_per_impl": used,
                        "wall_budget_s": None if a.steps else WALL_BUDGET_S,
                        "every": a.every, "resolution": a.res},
             "rows": allrows}, indent=2, default=str))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
