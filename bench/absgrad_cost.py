"""Is absgrad's 30% the sqrt in the inner loop, or a different splat layout?

Same scene, same gaussians, same backward -- only the absgrad buffer differs.

WARNING ABOUT THE FIRST VERSION OF THIS SCRIPT. It measured 2.5% and that
number was published as "the kernel costs 2.5%". It was not measuring the
kernel. The absgrad reduction, its sqrt and its tenth atomic ran
UNCONDITIONALLY in rasterize_backward; only the host-side accumulate was gated
on the buffer. Both arms therefore ran the identical kernel and the 2.5% was a
300k-element torch add. The kernel now takes a dispatch-uniform want_absgrad
flag, so the arms differ where this script claims they differ.
"""
import time, torch, sys
sys.path.insert(0, ".")
from metal_gauss import render

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--gaussians", type=int, default=300_000)
_ap.add_argument("--res", type=int, default=512)
_a = _ap.parse_args()

torch.manual_seed(0)
N, W, H = _a.gaussians, _a.res, _a.res
means = (torch.randn(N, 3) * 0.6 + torch.tensor([0., 0., 4.])).to("mps")
quats = torch.randn(N, 4).to("mps")
scales = (torch.rand(N, 3) * 0.05 + 0.01).to("mps")
opac = (torch.rand(N) * 0.7 + 0.15).to("mps")
cols = torch.rand(N, 3).to("mps")
K = torch.eye(3); f = 0.8 * max(W, H)
K[0,0], K[1,1], K[0,2], K[1,2] = f, f, W/2, H/2
vm = torch.eye(4)

def bench(use_buf, iters=30, warm=10):
    buf = torch.zeros(N, device="mps") if use_buf else None
    ts = []
    for i in range(iters + warm):
        m = means.clone().requires_grad_(True)
        if buf is not None:
            buf.zero_()
        torch.mps.synchronize(); t0 = time.perf_counter()
        rgb, _, _ = render(m, quats, scales, opac, None, K, vm, W, H,
                           colors=cols, backend="metal", absgrad_out=buf)
        rgb.square().mean().backward()
        torch.mps.synchronize()
        if i >= warm:
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts)//2] * 1000

# 2s sustained-load ramp: Apple Silicon boosts short bursts, and this repo has
# been fooled by that before.
t0 = time.perf_counter()
while time.perf_counter() - t0 < 2.0:
    m = means.clone().requires_grad_(True)
    r, _, _ = render(m, quats, scales, opac, None, K, vm, W, H, colors=cols,
                     backend="metal")
    r.square().mean().backward()
torch.mps.synchronize()

off = bench(False)
on = bench(True)
print(f"  fwd+bwd, {N:,} gaussians @ {W}x{H}")
print(f"    absgrad OFF : {off:7.2f} ms")
print(f"    absgrad ON  : {on:7.2f} ms   ({100*(on/off-1):+.1f}%)")
