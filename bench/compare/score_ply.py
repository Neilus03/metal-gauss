"""Score any 3DGS .ply on a scene's OFFICIAL test split, with one evaluator.

Why this exists: every trainer holds out its own views and computes PSNR with
its own code, so their reported numbers are not comparable to each other. On
NeRF-synthetic there IS an official test split, so the fair procedure is that
each implementation trains on the official train frames and exports a .ply, and
then a single evaluator renders the same 200 test views the same way.

The renderer used is our Metal path, which is validated against the
float64-gradcheck oracle (`metal_gauss.torch_ref`) to <=2e-3 on the forward.
`--oracle-check N` re-renders N views with the oracle and reports the largest
disagreement, so the claim "the evaluator does not favour our own output" is
something you can check rather than something you have to take on trust. The
oracle is not used for all 200 views only because it takes minutes per view.

Conventions must match metal_gauss/blender.py exactly: OpenGL camera-to-world
flipped to OpenCV world-to-camera, RGBA composited over WHITE. Getting either
wrong penalises every implementation equally but makes the absolute numbers
meaningless.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_test_views(scene: Path, max_side: int = 800):
    from PIL import Image
    meta = json.loads((scene / "transforms_test.json").read_text())
    angle_x = float(meta["camera_angle_x"])
    views = []
    for fr in meta["frames"]:
        p = scene / (fr["file_path"].lstrip("./") + ".png")
        if not p.exists():
            continue
        img = Image.open(p)
        scale = min(1.0, max_side / max(img.size))
        W, H = int(round(img.width * scale)), int(round(img.height * scale))
        if (W, H) != img.size:
            img = img.resize((W, H), Image.LANCZOS)
        a = np.asarray(img, dtype=np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4]) if a.shape[-1] == 4 else a[..., :3]
        focal = 0.5 * W / math.tan(0.5 * angle_x)
        K = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.0]],
                         dtype=torch.float32)
        c2w = np.array(fr["transform_matrix"], dtype=np.float32) @ _GL2CV
        vm = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))
        views.append((torch.from_numpy(rgb), K, vm, W, H))
    return views


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--scene", required=True, help="NeRF-synthetic scene dir")
    ap.add_argument("--resolution", type=int, default=800)
    ap.add_argument("--antialias", action="store_true",
                    help="score with the Mip-Splatting opacity compensation. "
                         "REQUIRED for a ply trained with --antialias: the "
                         "compensation is applied at render time and the ply "
                         "stores the RAW opacity, so scoring such a model "
                         "without this flag renders it with the wrong "
                         "rasteriser and understates it by ~2.5 dB.")
    ap.add_argument("--ssim", action="store_true",
                    help="also report SSIM (fused Metal, the same one the "
                         "training loss uses)")
    ap.add_argument("--lpips", action="store_true",
                    help="also report LPIPS (AlexNet). PSNR rewards blur, which "
                         "is the wrong bias when judging implementations that "
                         "ship antialiasing, so a perceptual metric matters "
                         "here even though it is slower.")
    ap.add_argument("--oracle-check", type=int, default=0,
                    help="re-render N views with torch_ref and report max disagreement")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from metal_gauss.io import load_ply
    from metal_gauss import render
    from metal_gauss.train import ssim as ssim_metal

    dev = "mps"
    sp = load_ply(a.ply, device=dev)
    views = load_test_views(Path(a.scene), a.resolution)
    if not views:
        raise SystemExit(f"no test views under {a.scene}")

    scores, ssims, lps = [], [], []
    kernel = lpips_net = None
    if a.ssim:
        from metal_gauss.train import _gaussian_kernel
        kernel = _gaussian_kernel(device=dev)
    if a.lpips:
        import lpips as _lpips
        lpips_net = _lpips.LPIPS(net="alex", verbose=False).to(dev)

    for gt, K, vm, W, H in views:
        with torch.no_grad():
            rgb, _, _ = render(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
                               K, vm, W, H, sh_degree=3, backend="metal",
                               background=(1.0, 1.0, 1.0), antialias=a.antialias)
        r = rgb.detach().clamp(0, 1)
        scores.append(psnr(r.cpu(), gt))
        if kernel is not None:
            # the fused Metal SSIM, i.e. the same one the training loss uses
            ssims.append(float(ssim_metal(r, gt.to(dev), kernel).item()))
        if lpips_net is not None:
            # LPIPS wants (N,3,H,W) in [-1,1]
            with torch.no_grad():
                x = r.permute(2, 0, 1)[None] * 2.0 - 1.0
                y = gt.to(dev).permute(2, 0, 1)[None] * 2.0 - 1.0
                lps.append(float(lpips_net(x, y).item()))

    mean = float(np.mean(scores))
    res = {"ply": a.ply, "scene": str(a.scene), "views": len(scores),
           "antialias": bool(a.antialias),
           "psnr": round(mean, 4), "n_splats": int(len(sp))}
    if ssims:
        res["ssim"] = round(float(np.mean(ssims)), 5)
    if lps:
        # lower is better, unlike PSNR and SSIM
        res["lpips"] = round(float(np.mean(lps)), 5)

    if a.oracle_check:
        worst = 0.0
        for gt, K, vm, W, H in views[:a.oracle_check]:
            with torch.no_grad():
                m, _, _ = render(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
                                 K, vm, W, H, sh_degree=3, backend="metal",
                                 background=(1.0, 1.0, 1.0))
                # torch_ref runs its projection in torch, so it needs the pose
                # on the gaussians' device; the Metal path wants it on the host.
                o, _, _ = render(sp.means, sp.quats, sp.scales, sp.opacities, sp.sh,
                                 K.to(dev), vm.to(dev), W, H, sh_degree=3,
                                 backend="torch_ref", background=(1.0, 1.0, 1.0))
            worst = max(worst, abs(psnr(m.cpu().clamp(0, 1), gt)
                                   - psnr(o.cpu().clamp(0, 1), gt)))
        res["oracle_psnr_disagreement"] = round(worst, 5)

    print(f"{Path(a.ply).name}: {mean:.4f} dB over {len(scores)} official test views, "
          f"{len(sp):,} splats"
          + (f"  [oracle disagrees by {res['oracle_psnr_disagreement']:.5f} dB]"
             if a.oracle_check else ""))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
