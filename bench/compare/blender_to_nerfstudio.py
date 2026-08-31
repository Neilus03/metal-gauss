"""NeRF-synthetic (Blender) -> Nerfstudio `transforms.json`.

msplat's loader auto-detects COLMAP (`cameras.bin`), Nerfstudio
(`transforms.json`) and Polycam only. NeRF-synthetic ships
`transforms_train.json` / `_test.json`, so it is invisible to msplat without
this conversion, and the Pareto front cannot include the synthetic scenes.

Conventions, and why this file does LESS than metal_gauss/blender.py:

  * `transform_matrix` is camera-to-world in OpenGL convention in BOTH formats,
    so it is copied through unchanged. Our own loader flips y and z because the
    RENDERER wants OpenCV world-to-camera; that flip belongs there, not here.
    Applying it twice mirrors the scene and still converges, a few dB down.
  * Blender stores only `camera_angle_x`; Nerfstudio wants explicit intrinsics.
    focal = 0.5 * W / tan(0.5 * angle_x), principal point at the centre.
  * `file_path` in Blender has no extension ("./train/r_0"); Nerfstudio wants a
    real path.

Only the official TRAIN frames are emitted. Test views are never handed to any
trainer -- they are held back for the common evaluator
(bench/compare/score_ply.py) so every implementation is scored on the same
views by the same code.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def convert(src: Path, dst: Path, split: str = "train", link: bool = True) -> dict:
    meta = json.loads((src / f"transforms_{split}.json").read_text())
    angle_x = float(meta["camera_angle_x"])

    from PIL import Image
    first = src / (meta["frames"][0]["file_path"].lstrip("./") + ".png")
    W, H = Image.open(first).size
    focal = 0.5 * W / math.tan(0.5 * angle_x)

    dst.mkdir(parents=True, exist_ok=True)
    img_dir = dst / "images"
    img_dir.mkdir(exist_ok=True)

    frames = []
    for fr in meta["frames"]:
        rel = fr["file_path"].lstrip("./") + ".png"
        srcp = src / rel
        if not srcp.exists():
            continue
        # Composite RGBA over WHITE and write RGB.
        #
        # Blender scenes ship transparent PNGs. Our loader composites over
        # white internally (the 3DGS convention), but handing the raw RGBA to
        # another trainer leaves it to guess: msplat evidently composited over
        # black, and the result was not merely a constant PSNR penalty -- its
        # training DIVERGED, PSNR falling 10.2 -> 7.0 -> 3.5 while the splat
        # count collapsed from 100k to 11k, because most of the target image
        # was uniform background. Pre-compositing gives every implementation
        # byte-identical inputs and removes the guess.
        outp = (img_dir / srcp.name).with_suffix(".png")
        if not outp.exists():
            im = Image.open(srcp)
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                bg.save(outp)
            else:
                shutil.copy2(srcp, outp)
        frames.append({
            "file_path": f"images/{srcp.name}",
            # OpenGL camera-to-world, copied through: Nerfstudio uses the same
            # convention Blender does.
            "transform_matrix": fr["transform_matrix"],
        })

    # NeRF-synthetic ships no sparse cloud, and msplat's Nerfstudio loader
    # looks for one -- without it, training starts from zero splats and does
    # nothing. Emit the SAME random init metal_gauss/blender.py generates (same
    # seed, same extent rule) so both trainers start from identical points and
    # the comparison is about the optimiser, not the initialisation.
    _write_sparse_ply(dst / "sparse.ply", frames, seed=0, n=100_000)

    out = {
        "camera_model": "OPENCV",
        "fl_x": focal, "fl_y": focal,
        "cx": W / 2.0, "cy": H / 2.0,
        "w": W, "h": H,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        # Required. Without it msplat loads the cameras, reports splats=0 and
        # trains on nothing -- silently, exit code 0. Dropping the file next to
        # transforms.json is not enough; the key is what it reads.
        "ply_file_path": "sparse.ply",
        "frames": frames,
    }
    (dst / "transforms.json").write_text(json.dumps(out, indent=2))
    return {"frames": len(frames), "W": W, "H": H, "focal": focal}


def _write_sparse_ply(path: Path, frames: list, seed: int = 0, n: int = 100_000) -> None:
    """Uniform random cube, sized from the camera ring -- mirrors load_blender()."""
    import numpy as np

    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    C = c2w[:, :3, 3]                                   # camera centres
    radius = float(np.linalg.norm(C - C.mean(0), axis=1).max())
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3)) - 0.5) * (radius * 0.6)
    cols = (rng.random((n, 3)) * 0.5 + 0.25) * 255.0

    with path.open("wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        arr["red"], arr["green"], arr["blue"] = (cols[:, 0].astype("u1"),
                                                 cols[:, 1].astype("u1"),
                                                 cols[:, 2].astype("u1"))
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="NeRF-synthetic scene dir (has transforms_train.json)")
    ap.add_argument("dst", help="output dir for transforms.json + images/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--copy", action="store_true", help="(kept for compatibility)")
    a = ap.parse_args()
    info = convert(Path(a.src), Path(a.dst), a.split, link=not a.copy)
    print(f"{a.src} -> {a.dst}: {info['frames']} frames, "
          f"{info['W']}x{info['H']}, focal {info['focal']:.2f}")


if __name__ == "__main__":
    main()
