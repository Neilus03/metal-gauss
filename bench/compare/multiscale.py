"""Does --antialias earn its place when render resolution != training resolution?

The 8-scene sweep found --antialias worth +0.01 dB on the mean: two wins, three
losses, three inside the noise floor. But that protocol trains at 800px and
scores at 800px, which is the regime where the Mip filter has least to do. Its
entire purpose is rendering at a resolution the model was not trained at, and
nothing in this repo has ever measured that.

So: train once per configuration, export the ply, and score the SAME ply at
several resolutions. score_ply.py LANCZOS-resizes the official test views, so
scoring at 400 compares a 400px render against a 400px ground truth -- which is
Mip-Splatting's own evaluation protocol.

If the compensation is doing what it claims, the gap should widen as the render
resolution moves away from the training resolution.
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


def score(ply: Path, scene: Path, res: int, antialias: bool) -> dict:
    # NOTE the asymmetry between the two filters. --antialias must be passed
    # here because it is applied at RENDER time and the ply stores raw
    # opacity. --filter-3d must NOT be, because it is baked into the exported
    # scales and opacity -- passing it would apply the filter twice. That
    # difference is the entire argument for preferring the 3D filter.
    """Score with the SAME rasteriser the model was trained under.

    --antialias applies its compensation at render time while the ply stores the
    RAW opacity, so scoring an antialias-trained model without the flag renders
    it with the wrong rasteriser. The first version of this script did exactly
    that and understated ficus by 2.5 dB, which looked like the technique
    failing rather than the harness mis-scoring it.
    """
    js = ply.with_suffix(f".{res}{'aa' if antialias else ''}.score.json")
    cmd = [sys.executable, str(ROOT / "bench" / "compare" / "score_ply.py"),
           str(ply), "--scene", str(scene), "--resolution", str(res),
           "--out", str(js)]
    if antialias:
        cmd.append("--antialias")
    subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not js.exists():
        return {"psnr": None}
    return json.loads(js.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["antialias", "filter-3d"],
                    default="antialias",
                    help="which filter to A/B against the unfiltered baseline")
    ap.add_argument("--scenes", nargs="*",
                    default=["ficus", "mic", "lego"],
                    help="default picks the biggest antialias win, biggest loss "
                         "and the reference scene")
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--train-res", type=int, default=800)
    ap.add_argument("--eval-res", nargs="*", type=int, default=[800, 400, 200])
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "multiscale.json"))
    a = ap.parse_args()

    outdir = Path("/tmp/cmp_out"); outdir.mkdir(exist_ok=True)
    rows, t0 = [], time.perf_counter()

    for scene in a.scenes:
        d = ROOT / "data" / "nerf_synthetic" / scene
        if not (d / "transforms_train.json").exists():
            print(f"  {scene}: MISSING"); continue
        for aa in (False, True):
            ply = outdir / f"ms_{scene}_{'aa' if aa else 'base'}.ply"
            spec = {"blender": str(d), "steps": a.steps,
                    "max_resolution": a.train_res,
                    "eval_every": a.steps * 10, "export": str(ply)}
            if aa:
                spec["antialias" if a.variant == "antialias" else "filter_3d"] = True
            try:
                rep = run(spec)
            except (RunFailed, RunDiverged) as e:
                print(f"  {scene} aa={aa}: FAILED {str(e)[:120]}")
                rows.append({"scene": scene, "antialias": aa, "ok": False,
                             "error": str(e)[:400]})
                continue
            # the flag must have actually reached the child
            key = "antialias" if a.variant == "antialias" else "filter_3d"
            assert rep["resolved"][key] == aa, \
                f"requested {key}={aa}, ran with {rep['resolved'][key]}"
            got = {}
            for res in a.eval_res:
                # baked filters need no scorer flag; render-time ones do
                need_flag = aa and a.variant == "antialias"
                got[res] = score(ply, d, res, need_flag)["psnr"]
            rows.append({"scene": scene, "antialias": aa, "ok": True,
                         "psnr_by_res": got, "resolved": rep["resolved"],
                         "env": rep["env"], "wall_s": rep["harness_wall_s"]})
            cells = "  ".join(f"{r}px {got[r]:.2f}" if got[r] else f"{r}px --"
                              for r in a.eval_res)
            print(f"  {scene:<10} antialias={str(aa):<5}  {cells}", flush=True)

    print("\n  delta (antialias - baseline), by render resolution:")
    print(f"    {'scene':<10}" + "".join(f"{r:>10}px" for r in a.eval_res))
    for scene in a.scenes:
        b = next((r for r in rows if r["scene"] == scene and not r["antialias"]
                  and r.get("ok")), None)
        w = next((r for r in rows if r["scene"] == scene and r["antialias"]
                  and r.get("ok")), None)
        if not b or not w:
            continue
        cells = ""
        for res in a.eval_res:
            pb, pw = b["psnr_by_res"].get(res), w["psnr_by_res"].get(res)
            cells += f"{pw - pb:>+11.2f}" if (pb and pw) else f"{'--':>11}"
        print(f"    {scene:<10}{cells}")

    Path(a.out).write_text(json.dumps(
        {"schema": 1, "env": _env(),
         "config": {"steps": a.steps, "train_res": a.train_res,
                    "eval_res": a.eval_res, "scenes": a.scenes},
         "total_wall_s": round(time.perf_counter() - t0, 1),
         "rows": rows}, indent=2, default=str))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
