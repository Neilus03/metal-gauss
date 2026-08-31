"""Re-score every point on the Pareto front with SSIM and LPIPS.

PSNR alone rewards blur, which is the wrong bias when the rival being dominated
(Brush) ships Mip-Splatting antialiasing and ours is off by default. If any
metric flatters us unfairly it is PSNR, so the front is re-scored on three.

No retraining: every ply behind the published front is still on disk.

THE STALE-PLY GUARD. /tmp/cmp_out holds exports from several runs with
overlapping names -- pareto_msplat_1000.ply and pareto_msplat_1000_auto.ply are
different files from different sweeps. Scoring the wrong one would silently
attach a correct-looking SSIM to the wrong model. So every candidate is
re-scored for PSNR FIRST and must reproduce the published value; the row is
only accepted from a ply that matches. If none matches, the row fails loudly
rather than being reported.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.provenance import env as _env  # noqa: E402

OUT_PLY = Path("/tmp/cmp_out")


def score(ply: Path, scene: Path, res: int, extra: list[str]) -> dict | None:
    js = ply.with_suffix(".metrics.json")
    cmd = [sys.executable, str(ROOT / "bench" / "compare" / "score_ply.py"),
           str(ply), "--scene", str(scene), "--resolution", str(res),
           "--out", str(js)] + extra
    subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return json.loads(js.read_text()) if js.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", default=str(ROOT / "bench" / "results" / "pareto_all.json"))
    ap.add_argument("--scene", default=str(ROOT / "data" / "nerf_synthetic" / "lego"))
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="dB tolerance when matching a ply to its published "
                         "row. Rendering the same ply is deterministic, so this "
                         "is tight on purpose -- it is an identity check, not a "
                         "noise allowance.")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "pareto_all_metrics.json"))
    a = ap.parse_args()

    front = json.loads(Path(a.front).read_text())
    rows, failures = [], []

    for r in front["rows"]:
        if not r.get("ok") or r.get("psnr") is None:
            rows.append(r); continue
        impl, iters, want = r["impl"], r["iters"], r["psnr"]
        # Try scene-qualified names first (current) then the older unqualified
        # ones, so previously-scored fronts still resolve. The PSNR identity
        # check below is what actually decides which file is right.
        sn = Path(a.scene).name
        cands = (sorted(OUT_PLY.glob(f"pareto_{sn}_{impl}_{iters}*.ply"))
                 + sorted(OUT_PLY.glob(f"pareto_{impl}_{iters}*.ply"),
                          key=lambda p: (0 if p.stem.endswith("_auto") else 1)))
        matched = None
        for ply in cands:
            got = score(ply, Path(a.scene), a.resolution, [])
            if got and abs(got["psnr"] - want) <= a.tol:
                matched = ply
                break
        if matched is None:
            tried = [f"{p.name}" for p in cands]
            print(f"  {impl:<12} {iters:>6}  NO PLY REPRODUCES {want:.3f} dB "
                  f"(tried {tried})", flush=True)
            failures.append({"impl": impl, "iters": iters, "want": want,
                             "tried": tried})
            rows.append(r); continue

        full = score(matched, Path(a.scene), a.resolution, ["--ssim", "--lpips"])
        out = dict(r)
        out.update({"ssim": full.get("ssim"), "lpips": full.get("lpips"),
                    "ply": matched.name, "psnr_recheck": full.get("psnr")})
        rows.append(out)
        print(f"  {impl:<12} {iters:>6}  PSNR {full['psnr']:6.2f}  "
              f"SSIM {full.get('ssim', float('nan')):.4f}  "
              f"LPIPS {full.get('lpips', float('nan')):.4f}   [{matched.name}]",
              flush=True)
        Path(a.out).write_text(json.dumps(
            {"schema": 1, "env": _env(),
             "config": {"scene": Path(a.scene).name, "resolution": a.resolution,
                        "metrics": ["psnr", "ssim", "lpips"], "tol_db": a.tol},
             "failures": failures, "rows": rows}, indent=2, default=str))

    print(f"\n  {sum(1 for r in rows if r.get('ssim') is not None)} rows re-scored, "
          f"{len(failures)} could not be matched to a ply")
    print(f"-> {a.out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
