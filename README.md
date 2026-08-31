<p align="center"><img src="assets/logo.png" width="320"></p>

<p align="center"><b>3D Gaussian Splatting that trains on Apple Silicon. Metal kernels, no CUDA, no Xcode.</b></p>

<p align="center">
  <img alt="Apple Silicon" src="https://img.shields.io/badge/platform-Apple%20Silicon-000000?logo=apple&logoColor=white">
  <img alt="Metal" src="https://img.shields.io/badge/backend-Metal-A855F7">
  <img alt="PyTorch MPS" src="https://img.shields.io/badge/PyTorch-MPS-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-22C55E">
  <img alt="No CUDA" src="https://img.shields.io/badge/CUDA-not%20required-6E7681">
  <img alt="No Xcode" src="https://img.shields.io/badge/Xcode-not%20required-6E7681">
</p>

---

3D Gaussian Splatting that **trains** on Apple Silicon. Metal compute kernels compiled at runtime,
so there is no CUDA and no Xcode — Command Line Tools are enough.

## 🏆 Quality per minute

Given N minutes, what is the best reconstruction you can get? All 8 NeRF-synthetic scenes, identical
seed cloud per scene, one evaluator on the official 200-view test split, strictly sequential.

<!-- BEGIN:budget -->
| you have | metal-gauss | msplat | Brush | spirula |
|---|---:|---:|---:|---:|
| 30 s | **21.5** | 19.9 | 12.8 | 13.8 |
| 1 min | **24.2** | 20.8 | 14.4 | 15.1 |
| 3 min | **29.8** | 22.1 | 18.4 | 17.4 |
| 6 min | **31.4** | 22.3 | 23.4 | 19.6 |
| 15 min | **31.9** | 22.4 | 26.9 | 22.1 |
| 30 min | **31.9** | 22.4 | 26.9 | 28.5 |

*Best 8-scene-mean PSNR reachable without exceeding each budget; msplat takes its better variant at each point. Em dash means the implementation produces nothing within that budget on all 8 scenes. Full per-rung ladder in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).*
<!-- END:budget -->

![PSNR vs wall-clock, 8-scene mean](bench/results/pareto_8scene.svg)

Lines trace best-achievable-by-budget. Hollow dots are measured but beaten by a cheaper run of the
same implementation. Whiskers are ±1 s.e.m. across the 8 scenes.

![Paired per-scene margin with 95% confidence intervals](bench/results/margin_forest.svg)

Paired per scene, because every implementation ran the same 8 scenes. One interval crosses zero:
our quality margin over spirula at 15 k is not resolved by 8 scenes, though the wall-clock margin
there (6.0 min vs 28.4) is not in question.

Domination (faster **and** better) on PSNR / on PSNR+SSIM:
[Brush](https://github.com/ArthurBrussee/brush) **48/48 · 48/48**,
[spirula-studio](https://github.com/harry7557558/spirula-studio) **45/48 · 41/48**,
[msplat](https://github.com/rayanht/msplat) **66/96 · 50/96**.

## ⏱️ Watch it converge

Four trainers, same seed cloud, each given the **same ~390 s**, running however many iterations fit.

![Side-by-side convergence against wall-clock](assets/timelapse.gif)

| | final PSNR | first 20 dB | first 24 dB | first 27 dB |
|---|---:|---:|---:|---:|
| **metal-gauss** (15 k it) | **33.26** | 19.4 s | **57.5 s** | **147.1 s** |
| Brush (7 k it) | 24.82 | 105.6 s | 270.0 s | — |
| spirula-studio (5.5 k it) | 24.31 | 147.8 s | 371.6 s | — |
| msplat (19 k it) | 23.33 | **7.9 s** | 58.5 s | — |

lego, panel at 400 px, metrics over 20 held-out views at 800 px. Build the interactive version with
`python bench/compare/build_timelapse_page.py`.

## 📦 Install

```bash
pip install "git+https://github.com/nandometzger/metal-gauss"
```

macOS on Apple Silicon, Python ≥3.10, PyTorch ≥2.5. Metal kernels compile at **runtime** — no Xcode,
no `.metallib` step.

## 🚀 Train

```bash
metal-gauss-train --colmap scene/sparse/0 --images scene/images --steps 7000 --export scene.ply
metal-gauss-train --blender data/nerf_synthetic/lego --steps 30000
```

Every 8th view is held out. The exported `.ply` is standard INRIA-convention 3DGS. Capacity follows
the step count; `--budget` overrides it. Blender scenes are **not vendored** — unpack
[nerf_synthetic](https://github.com/bmild/nerf) into `data/nerf_synthetic/`.

## ⚠️ Caveats

- msplat is **1.3–1.8× faster per step** and owns every budget under ~0.3 min. Our floor there is
  ~8.4 s of fixed startup.
- 7 k numbers are **not comparable to published 30 k numbers** — 5.1 dB apart.
- `--antialias` is off by default; worth **+6.68 dB at 200 px** render resolution.
- Run-to-run noise floors: ours **0.19 dB**, Brush **0.74**, spirula **0.15–1.27**, msplat **3.35**
  (no seed flag).
- `--budget` at 1 M splats: **+2.75 dB** lego, **+1.40** mic, **−0.20** ship, all at ~5× the time.

## 📚 More

| | |
|---|---|
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | full results, protocol, calibration, noise floors |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it works, the kernels, correctness oracle |
| [bench/results/NEGATIVE_RESULTS.md](bench/results/NEGATIVE_RESULTS.md) | **every rejected lever and measurement lesson, with numbers** |
| [bench/compare/STATUS.md](bench/compare/STATUS.md) | every Apple-native implementation surveyed, and the traps |

`NEGATIVE_RESULTS.md` is the most useful file here: several published speedups measure near zero on
this hardware, and several of this repo's own conclusions were wrong until re-measured.

## 📄 License

MIT. Credits: [3DGS-MCMC](https://arxiv.org/abs/2404.09591),
[Taming 3DGS](https://arxiv.org/abs/2406.15643), [Speedy-Splat](https://arxiv.org/abs/2412.00578),
[LiteGS](https://arxiv.org/abs/2503.01199), [Mip-Splatting](https://arxiv.org/abs/2311.16493),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio),
[Brush](https://github.com/ArthurBrussee/brush), and the original
[INRIA rasterizer](https://github.com/graphdeco-inria/gaussian-splatting).
