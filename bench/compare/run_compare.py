"""Head-to-head on one COLMAP scene, same machine, same iteration count.

Each implementation is scored by ITS OWN eval path where it has one. We do not
re-score another project's .ply with our metric -- that would quietly advantage
whichever renderer the metric was written against. Where a project reports no
held-out PSNR, the row carries speed only and says so.

Nothing here runs concurrently: GPU contention is the single easiest way to
produce a wrong benchmark, and this machine's timings already swing 40% between
boost and sustained clock (see bench/results/NEGATIVE_RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"

# Every implementation reports PSNR differently; these are deliberately narrow
# so a format change shows up as "no psnr" rather than a silently wrong number.
PSNR_PATTERNS = [
    # metal-gauss no longer reaches this path -- it goes through
    # bench/runner.py and reads its own report. Kept only so a stray
    # metal-gauss log is still parsed rather than silently scored as None.
    r"heldout PSNR\s+([0-9.]+)",
    r"[Pp]SNR[:= ]+([0-9.]+)",            # msplat, splat-apple
    r"psnr\s*=\s*([0-9.]+)",
]


def parse_psnr(text: str):
    best = None
    for pat in PSNR_PATTERNS:
        for m in re.finditer(pat, text):
            try:
                best = float(m.group(1))       # last match = final eval
            except ValueError:
                pass
    return best


def run_ours(name: str, spec: dict) -> dict:
    """metal-gauss goes through bench/runner.py, never through stdout.

    The competitors below have no machine-readable report, so scraping their
    output is the only channel available for them and is guarded by the
    exit-0-is-not-success check in run(). We do have a report, so declining to
    use it would be choosing the weaker channel for the one implementation
    where the stronger one exists -- and stdout scraping is what turned
    "278,571 splats" into 571.
    """
    from bench.runner import RunDiverged, RunFailed
    from bench.runner import run as run_trainer
    print(f"\n=== {name} ===\n$ spec {spec}", flush=True)
    t0 = time.perf_counter()
    try:
        rep = run_trainer(spec)
    except (RunFailed, RunDiverged) as e:
        print(f"  FAILED: {str(e)[:300]}", flush=True)
        return {"impl": name, "ok": False, "returncode": None,
                "wall_s": round(time.perf_counter() - t0, 1),
                "psnr": None, "cmd": str(spec), "tail": str(e)[:1000]}
    m = rep["metrics"]
    print(f"  wall={m['wall_s']/60:.1f} min  psnr={m['psnr']}  "
          f"budget={rep['resolved']['budget']:,}", flush=True)
    return {"impl": name, "ok": m["psnr"] is not None, "returncode": 0,
            "wall_s": m["wall_s"], "psnr": m["psnr"],
            "n_splats": m["n_splats"], "resolved": rep["resolved"],
            "env": rep["env"], "cmd": " ".join(rep["cmd"]), "tail": ""}


def run(name: str, cmd: list[str], cwd=None, env=None) -> dict:
    print(f"\n=== {name} ===\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       cwd=cwd, env=env)
    wall = time.perf_counter() - t0
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    psnr = parse_psnr(out)
    # A zero exit code is not proof of a run. splat-apple prints
    # "Error loading colmap data" and returns 0, which would land in the table
    # as a successful row with a missing PSNR. Treat an implausibly fast run,
    # or visible error text, as failure.
    err_marker = re.search(r"(?i)\b(error|traceback|not implemented|failed)\b", out)
    ok = p.returncode == 0 and wall > 5.0 and not err_marker
    print(f"  rc={p.returncode}  wall={wall/60:.1f} min  psnr={psnr}", flush=True)
    if not ok:
        print("  tail: " + " | ".join(out.strip().splitlines()[-3:])[:300], flush=True)
    if p.returncode == 0 and not ok:
        print(f"  NOTE: exit 0 but treated as failure "
              f"({'error text in output' if err_marker else 'finished implausibly fast'})",
              flush=True)
    return {"impl": name, "ok": ok, "returncode": p.returncode,
            "wall_s": round(wall, 1), "psnr": psnr,
            "cmd": " ".join(str(c) for c in cmd),
            "tail": "\n".join(out.strip().splitlines()[-15:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="/tmp/cmp_data/room1",
                    help="standard COLMAP layout: images/ and sparse/0/")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--budget", type=int, default=600_000,
                    help="matched capacity for the head-to-head; the resolved "
                         "value is recorded in every row")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(RESULTS / "head_to_head.json"))
    args = ap.parse_args()

    scene = Path(args.scene)
    rows = []

    jobs = {
        "metal-gauss": lambda: run_ours("metal-gauss", {
            "colmap": scene / "sparse" / "0", "images": scene / "images",
            "steps": args.iters, "budget": args.budget,
            "max_resolution": 1600, "eval_every": args.iters}),
        "msplat": lambda: run("msplat", [
            "/tmp/cmp_msplat/bin/msplat-train", "--input", scene,
            # keep the input coordinate frame, so an external evaluator can
            # score the exported .ply -- see bench/compare/STATUS.md
            "--num-iters", args.iters, "--keep-crs", "--eval",
            "--output", "/tmp/cmp_out/msplat.ply"]),
        "splat-apple-mlx": lambda: run("splat-apple-mlx", [
            "/tmp/cmp_sa/bin/python", "train_mlx.py",
            "--data_dir", scene, "--img_folder", "images",
            "--num_iterations", args.iters,
            "--rasterizer", "cpp"], cwd="/tmp/cmp/splat-apple"),
        "splat-apple-torch": lambda: run("splat-apple-torch", [
            "/tmp/cmp_sa/bin/python", "train_torch.py",
            "--data_dir", scene, "--img_folder", "images",
            "--num_iterations", args.iters, "--device", "mps",
            "--rasterizer", "cpp"], cwd="/tmp/cmp/splat-apple"),
        "opensplat": lambda: run("opensplat", [
            "/tmp/cmp/OpenSplat/build/opensplat", scene,
            "-n", args.iters, "-o", "/tmp/cmp_out/opensplat.ply", "--val"]),
    }

    Path("/tmp/cmp_out").mkdir(exist_ok=True)
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        try:
            rows.append(fn())
        except Exception as e:                      # noqa: BLE001 - record, continue
            print(f"  {name}: harness error {e}", flush=True)
            rows.append({"impl": name, "ok": False, "error": str(e)})

    out = Path(args.out)
    out.write_text(json.dumps(
        {"scene": str(scene), "iters": args.iters, "rows": rows}, indent=2))
    print(f"\n{'impl':<20}{'ok':<5}{'wall (min)':>12}{'PSNR':>9}")
    for r in rows:
        w = f"{r['wall_s']/60:.1f}" if r.get("wall_s") else "-"
        q = f"{r['psnr']:.2f}" if r.get("psnr") else "-"
        print(f"{r['impl']:<20}{str(r.get('ok')):<5}{w:>12}{q:>9}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
