"""Frames per second for playback rendering, forward only.

Distinct from `bench/quick.py`, which times the training step: that number is
dominated by the backward, and the backward is not run here. What a viewer or a
`metal-gauss-render` job actually costs is the forward alone, so it gets its own
measurement rather than an estimate derived from the training figure.

Timing discipline is copied from `bench/quick.py` and is not optional. Apple
Silicon runs short bursts at a boost clock and settles lower under sustained
load, so the same kernel measures 11.6 or 19.8 ms depending on nothing but how
long it has been running. Rendering a video IS sustained load, so the settled
clock is the honest number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metal_gauss import render as raster
from metal_gauss.io import load_ply
from metal_gauss.render_path import camera_path, intrinsics


def timed(fn, k: int = 11, ramp_s: float = 2.0) -> float:
    """Trimmed median of k, in ms, after ramping the GPU to steady state."""
    end = time.perf_counter() + ramp_s
    while time.perf_counter() < end:
        fn()
    torch.mps.synchronize()
    fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(sorted(ts)[1:-1])) * 1000.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ply")
    ap.add_argument("--counts", type=int, nargs="+", default=[100_000, 300_000, 600_000])
    ap.add_argument("--resolutions", type=int, nargs="+", default=[256, 384, 512, 768])
    ap.add_argument("--repeats", type=int, default=3,
                    help="round-robin sweeps over every configuration")
    ap.add_argument("--ramp", type=float, default=2.0,
                    help="seconds of continuous work before each timed sample")
    ap.add_argument("--inner", type=int, default=9,
                    help="timed frames per sample, trimmed median")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--allow-contended", action="store_true",
                    help="skip the exclusivity check; the numbers then mean nothing")
    a = ap.parse_args(argv)

    if not a.allow_contended:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from runner import require_gpu_exclusive
        require_gpu_exclusive()

    sp = load_ply(a.ply, device="mps")
    print(f"{len(sp)} splats, SH degree {sp.sh_degree}\n")

    # One fixed view. The path shape does not change the cost; the splat count
    # and the pixel count do.
    vm = camera_path([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1, 0.0)[0]

    # A contiguous prefix, not a random subset: gather would add a cost the
    # renderer does not have.
    configs = [(n, res) for n in a.counts if n <= len(sp) for res in a.resolutions]
    subsets = {n: sp.subset(slice(0, n)) for n in a.counts if n <= len(sp)}

    def make(n, res):
        s, K = subsets[n], intrinsics(res, res, 45.0)

        def fwd():
            raster(s.means, s.quats, s.scales, s.opacities, s.sh,
                   K, vm, res, res, sh_degree=sp.sh_degree, backend="metal")
        return fwd

    # Sweep every configuration `repeats` times round-robin rather than
    # finishing one before starting the next. Measured in a fixed order, slow
    # drift over the session is indistinguishable from a property of whichever
    # config ran late: a first pass here had 256px looking SLOWER than 384px,
    # and re-running with the order changed moved the number by 2x. Round-robin
    # spreads any drift across all of them, and the spread across repeats is
    # reported so a contended run is visible rather than silently averaged in.
    samples: dict[tuple[int, int], list[float]] = {c: [] for c in configs}
    for r in range(a.repeats):
        for c in configs:
            # Full ramp on EVERY sample, not just the first. Shortening it for
            # later repeats to save wall-clock let the clock drop between
            # configurations and pushed the spread past 150%, which is the
            # same mistake as not ramping at all, just harder to see.
            samples[c].append(timed(make(*c), k=a.inner, ramp_s=a.ramp))

    rows = []
    print(f"  {'splats':>9}  {'res':>9}  {'ms/frame':>9}  {'fps':>7}  {'agree':>7}")
    for (n, res) in configs:
        v = sorted(samples[(n, res)])
        ms = float(np.median(v))
        # Spread of the repeats EXCLUDING the single worst, not the full range.
        # One interruption on a 1.7 ms configuration moves the range by 140%
        # while leaving the median untouched, so gating on the range condemns
        # measurements that are fine. This asks the question that actually
        # matters: do the repeats agree once one hiccup is set aside.
        agree = (v[-2] - v[0]) / ms * 100.0 if len(v) > 1 and ms else 0.0
        rows.append({"splats": n, "res": res, "ms": round(ms, 2),
                     "fps": round(1000.0 / ms, 1),
                     "agree_pct": round(agree, 1),
                     "range_pct": round((v[-1] - v[0]) / ms * 100.0, 1) if ms else 0.0,
                     "samples": [round(x, 2) for x in v]})
        print(f"  {n:>9,}  {res:>4}x{res:<4}  {ms:>9.2f}  {1000.0/ms:>7.1f}  {agree:>6.1f}%")
    worst = max(r["agree_pct"] for r in rows) if rows else 0.0
    if worst > 15.0:
        print(f"\n  WARNING: repeats disagree by {worst:.0f}% even after discarding "
              f"the worst sample. Something else was using the GPU; these numbers "
              f"do not mean anything.")
    else:
        print(f"\n  repeats agree to within {worst:.0f}% (worst case, "
              f"discarding one outlier per configuration)")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
