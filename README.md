# VIPE to Gaussian Splatting

Reproducible pipeline that turns an ordered set of drone stills into a Gaussian Splatting model and a
rendered camera-path video, using NVIDIA ViPE for camera poses, intrinsics and metric depth, and
Nerfstudio Splatfacto for the splats.

```
stills ──▶ prepare ──▶ ViPE ──▶ COLMAP export ──▶ Splatfacto ──▶ PLY + fly-through
```

This repository contains the scripts and the configuration only. Datasets, environments, checkpoints,
exports and videos are produced by running it and are written to the network volume — nothing
generated is committed here.

Every stage has a verification step that fails loudly rather than passing bad data downstream. One of
those checks found a camera pose that ViPE reported wrong without any warning; see
[Known deviation](#known-deviation-from-a-pure-vipe-solve).

- **How to run it:** [docs/RUNBOOK.md](docs/RUNBOOK.md) — every command in order, with timings
- **What each stage guarantees:** [docs/PIPELINE.md](docs/PIPELINE.md)
- **Environment that was verified:** [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- **Defects hit while building this, and how each was diagnosed:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Requirements

- Linux host with an NVIDIA GPU of at least 24 GB; the reference run peaked at 24,806 MiB
- CUDA driver new enough for the GPU architecture in use
- Persistent storage mounted at `/workspace` — the pipeline installs its environments there so they
  survive host replacement
- `git`, `curl`, `ffmpeg`, `ffprobe`. Conda is **not** required in advance: the setup installs a
  pinned Miniforge itself

## Quick start

```bash
bash scripts/verify_environment.sh host
bash scripts/verify_dataset.sh
bash scripts/setup_vipe.sh
bash scripts/setup_splatfacto.sh

python3 scripts/prepare_dataset.py \
  --input /workspace/datasets/<archive>.zip \
  --output /workspace/gs_test/data/prepared/<name> \
  --fps 1 --max-width 1920

bash scripts/run_vipe.sh
# verify the solve before paying for training — see the runbook
bash scripts/prepare_colmap.sh
bash scripts/train_splatfacto.sh
bash scripts/eval_splatfacto.sh
bash scripts/export_splat.sh

python3 scripts/make_camera_path.py \
  --config <trained run>/config.yml \
  --output /workspace/gs_test/artifacts/camera_path.json
bash scripts/render_camera_path.sh /workspace/gs_test/artifacts/camera_path.json
```

For a cheap trial pass, `prepare_dataset.py --limit 20` builds a short sequence and
`SPLATFACTO_MAX_ITERATIONS=2000` keeps the first training run brief.

## Configuration

Dependency pins live in [config/versions.env](config/versions.env): the ViPE and Nerfstudio commits by
40-character SHA, the CUDA, torch, gsplat and numpy versions, the Miniforge release and the pod image
tag. Runtime paths are parameterised in [config/paths.env.example](config/paths.env.example); copy it
to `config/paths.env` only if a machine needs overrides.

Version pins are hard assignments. Run parameters — frame count, fps, iterations, eval interval, seed
— use the `${VAR:-default}` form, so they can be overridden from the environment without editing the
manifest:

```bash
SPLATFACTO_MAX_ITERATIONS=2000 bash scripts/train_splatfacto.sh
```

## Why two environments

ViPE and Nerfstudio cannot share one. Nerfstudio 1.1.5 pins `viser==0.2.7` where ViPE needs 1.0.27,
`timm` 0.6.7 against 1.0.27, and `protobuf` 3.20.3 against 7.34.1 — 45 of their 91 shared packages
differ. ViPE also installs through `uv sync --frozen`, which removes anything pip adds behind its
back, so the two installers would undo each other.

So `setup_vipe.sh` builds a conda prefix carrying the CUDA toolchain plus a uv-managed virtualenv for
ViPE's Python dependencies, and `setup_splatfacto.sh` builds a second, independent conda prefix for
Nerfstudio. Both live on the volume.

## Reference run

Measured on an NVIDIA RTX PRO 4500 Blackwell (32 GB), 126 stills at 1920×1440 captured at 1 fps.
These numbers describe that run; the outputs themselves are not part of this repository.

| Stage | Outcome |
| --- | --- |
| Dataset check | 126 frames, every one checksum-matched to the source archive |
| ViPE | 126 poses in 353 s, peak 24,806 MiB |
| Splatfacto | 30,000 iterations in 43 min 50 s, peak 16,694 MiB, 7,239,301 gaussians |
| Export | 7,055,547 gaussians in a 1.75 GB PLY with spherical harmonics |
| Render | 625 frames, 1920×1440, h264, 30 fps, 20.83 s |
| Total | about 1 h 15 min of GPU time |

Quality from `ns-eval` on the 16 frames held out of optimization (every 8th):

| Metric | Held-out frames | Training frames |
| --- | --- | --- |
| PSNR | 15.13 (σ 2.54) | 22.29 median |
| SSIM | 0.326 (σ 0.14) | — |
| LPIPS | 0.454 (σ 0.09) | — |

This is a modest result and worth reading honestly. The ~7 dB gap between fitted and held-out views
means the model reconstructs the scene it saw and generalises poorly to new viewpoints. Renders are
recognisable — building, roof, trees and snow in the right places — but show streaky elongated
gaussians and misregistration against the reference.

Likely causes, in order: scene content (leafless vegetation is high-frequency and view-dependent, and
PSNR punishes every misplaced branch), the sparse-view regime of 110 training views over a several
hundred metre flight, large baselines from 1 fps capture, and residual pose error — the 95th
percentile disagreement with image-derived rotations is 1.64°, which at `fx ≈ 1435` is tens of pixels.

Untried levers, in order of expected gain: Splatfacto's camera optimizer
(`--pipeline.model.camera-optimizer.mode SO3xR3`), per-image exposure embeddings (DJI auto-exposure
drifts along a flight), and depth supervision from ViPE's metric depth, which this pipeline already
exports.

ViPE itself is accurate on this data: across 122 usable consecutive pairs its rotations agree with
SIFT/essential-matrix estimates recovered from the images to a median of 0.26° and a 95th percentile
of 1.64°.

## Known deviation from a pure ViPE solve

On the reference dataset, one of the 125 relative camera poses came out wrong. ViPE solved 64.74° of
rotation between frames 22 and 23, where an independent essential-matrix estimate from 867 SIFT
matches gives 4.64°. Four ViPE configurations — default, `no_vda`, a huber robust kernel, and raised
SLAM thresholds — moved that defect around frames 22–25 but never removed it, and ViPE logged no
warning about any of it.

`scripts/repair_pose.py` re-solves that single relative pose with RGB-D PnP from ViPE's own metric
depth, then carries the rigid correction into the later poses and into the SLAM points contributed by
the shifted frames, which would otherwise be left behind their cameras.

The repair refuses to run unless it first reproduces a pair the solver got right, and it determines
the pose convention from the data instead of assuming one. On the reference run it took the pair from
64.74° to 7.33° and its step from 36.47 to 5.86; afterwards the cross-check reported zero disagreeing
pairs, a worst error of 3.37° instead of 60.79°, and an unchanged median — the repair touched the
defect and nothing else.

This step is conditional. Run the cross-check first; if it reports no disagreement, skip the repair.

## Repository layout

```text
config/      version pins and the machine-specific path template
docs/        runbook, stage contracts, verified environment, defect log
scripts/     setup, execution and verification entrypoints
scripts/lib/ shared shell utilities
```

## Upstream projects

- [NVIDIA ViPE](https://github.com/nv-tlabs/vipe) — poses, intrinsics, metric depth and the SLAM map,
  pinned by commit; its `scripts/vipe_to_colmap.py` produces the COLMAP export
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) — Splatfacto training, evaluation,
  export and rendering
- [gsplat](https://github.com/nerfstudio-project/gsplat) — the CUDA rasteriser behind Splatfacto
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) and UniDepth — depth
  priors ViPE downloads at runtime
- [Miniforge](https://github.com/conda-forge/miniforge) and [uv](https://github.com/astral-sh/uv) —
  environment and dependency management
