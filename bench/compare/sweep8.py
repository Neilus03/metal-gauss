"""The full NeRF-synthetic competitor sweep: 8 scenes x 4 implementations.

Ordering and resumability are the whole design here, because this runs for
roughly half a day and a machine that sleeps mid-run must not cost the lot.

  * Scene order puts lego, drums and ficus FIRST, so a partial run still
    covers the three scenes with the longest measurement history.
  * msplat runs BOTH variants everywhere. Neither is uniformly fairer: stock is
    far faster because it stays at quarter resolution longer and wins on ficus
    by +4.50 dB at 3.3x the speed, while scaled wins the short rungs on lego by
    up to +5.12 dB. Reporting one fixed choice understates it either way, so
    both are measured and the front is taken over the union.
  * Each (scene, impl, variant) writes its own JSON and is skipped if that file
    already exists, so re-invoking resumes rather than restarts.

Wall-clock rows are only meaningful on an idle GPU, so this refuses to start if
another trainer is running and never runs two of its own at once.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.runner import require_gpu_exclusive  # noqa: E402

RESULTS = ROOT / "bench" / "results" / "sweep8"
# Published-claim scenes first; see module docstring.
SCENES = ["lego", "drums", "ficus", "chair", "hotdog", "materials", "mic", "ship"]


def run(cmd: list[str], log: Path) -> tuple[bool, float]:
    t0 = time.perf_counter()
    with log.open("w") as fh:
        p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                           stdout=fh, stderr=subprocess.STDOUT, text=True)
    return p.returncode == 0, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--impls", nargs="*",
                    default=["metal-gauss", "msplat-stock", "msplat-scaled",
                             "brush", "spirula"])
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    require_gpu_exclusive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "logs").mkdir(exist_ok=True)
    todo, done_already = [], []
    for scene in a.scenes:
        for impl in a.impls:
            out = RESULTS / f"{scene}_{impl}.json"
            (done_already if out.exists() else todo).append((scene, impl, out))

    print(f"{len(todo)} to run, {len(done_already)} already present", flush=True)
    if a.dry_run:
        for scene, impl, _ in todo:
            print(f"  would run {scene:<10} {impl}")
        return

    t_start = time.perf_counter()
    for i, (scene, impl, out) in enumerate(todo, 1):
        ns = f"/tmp/cmp_data/{scene}_ns"
        blender = ROOT / "data" / "nerf_synthetic" / scene
        log = RESULTS / "logs" / f"{scene}_{impl}.log"
        if impl == "spirula":
            cmd = [sys.executable, ROOT / "bench/compare/run_spirula.py",
                   "--scene", blender, "--data", ns,
                   "--resolution", a.resolution, "--out", out]
        else:
            base = [sys.executable, ROOT / "bench/compare/pareto.py",
                    "--scene", blender, "--scene-ns", ns,
                    "--resolution", a.resolution, "--out", out]
            if impl == "metal-gauss":
                cmd = base + ["--impls", "metal-gauss"]
            elif impl == "msplat-stock":
                cmd = base + ["--impls", "msplat", "--msplat-stock"]
            elif impl == "msplat-scaled":
                cmd = base + ["--impls", "msplat"]
            elif impl == "brush":
                cmd = base + ["--impls", "brush"]
            else:
                print(f"  unknown impl {impl}, skipped", flush=True)
                continue

        print(f"[{i}/{len(todo)}] {scene:<10} {impl:<14} ...", end="", flush=True)
        ok, wall = run(cmd, log)
        el = time.perf_counter() - t_start
        rate = el / i
        print(f" {'ok ' if ok else 'FAILED'} {wall/60:6.1f} min"
              f"   | elapsed {el/3600:.2f} h, eta {(len(todo)-i)*rate/3600:.2f} h",
              flush=True)
        if not ok:
            # Do not abort the sweep: one implementation failing on one scene
            # is a recorded result, not a reason to lose the other 39 cells.
            print(f"      see {log}", flush=True)

    print(f"\ndone in {(time.perf_counter()-t_start)/3600:.2f} h -> {RESULTS}")


if __name__ == "__main__":
    main()
