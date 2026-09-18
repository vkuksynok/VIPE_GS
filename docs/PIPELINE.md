# Pipeline contracts

All commands are executed from the repository root on a Linux host with an NVIDIA GPU. Version pins come from `config/versions.env`; paths come from `config/paths.env.example` plus optional `config/paths.env` overrides.

| Stage | Entrypoint | Required input | Produced output | Completion check |
| --- | --- | --- | --- | --- |
| Host verification | `bash scripts/verify_environment.sh host` | NVIDIA GPU pod with `/workspace` | environment report under `artifacts/environment/` | CUDA is available; `/workspace` is mounted and writable |
| Dataset verification | `bash scripts/verify_dataset.sh` | source archives on the network volume | runtime directory tree, `data/raw/<archive>/`, JSON report under `artifacts/environment/` | archive SHA-256 recorded, ZIP integrity clean, 126 frames fully decoded with contiguous sequence numbers and no duplicate names |
| VIPE setup | `bash scripts/setup_vipe.sh` | Git, Conda, CUDA-capable host | pinned checkout, Conda native env, frozen uv env | reported Git SHA equals manifest; `vipe --help` succeeds |
| Splatfacto setup | `bash scripts/setup_splatfacto.sh` | Git, Conda, compatible NVIDIA driver | pinned Nerfstudio checkout and isolated Conda env | exact torch, CUDA, Nerfstudio and gsplat versions import successfully |
| Dataset preparation | `python scripts/prepare_dataset.py ...` | ZIP or directory with ordered DJI JPEGs; FFmpeg | `images/`, `frames.csv`, `manifest.json`, MP4 | 126 unique source frames map to 126 encoded frames |
| Sequence check | `python scripts/verify_prepared_sequence.py --prepared-dir ... --dataset-report ...` | a prepared sequence and the dataset report | JSON summary under `artifacts/environment/` | contiguous indices in capture order, images and CSV agree, ffprobe frame count and aspect ratio match, every frame traces to its source photo by SHA-256 |
| VIPE inference | `bash scripts/run_vipe.sh` | prepared MP4 and VIPE env | VIPE artifacts, camera poses, intrinsics, depth and SLAM map | process exits 0; required artifact groups and log exist |
| VIPE output check | `python scripts/verify_vipe_outputs.py --output-dir ... --sequence ...` | a completed VIPE run | JSON summary under `artifacts/environment/` | every artifact group and the SLAM map exist; poses are finite with orthonormal rotations and real motion; intrinsics are positive and match the input resolution |
| Rotation cross-check | `python scripts/crosscheck_pose_rotations.py --images ... --poses ... --intrinsics ...` | prepared frames and solved poses | JSON summary under `artifacts/environment/` | solved rotations agree with SIFT/essential-matrix rotations recovered from the images; pairs with too few inliers count as unusable, not as disagreements |
| Pose repair (only when the cross-check finds a bad pair) | `python scripts/repair_pose.py --pair I --validate-pair J ...` | poses, intrinsics, depth archive, frames | corrected poses and SLAM map | the procedure first reproduces a known-good pair, then the repaired pair agrees with the images and every later relative pose is unchanged |
| COLMAP conversion | `bash scripts/prepare_colmap.sh` | VIPE artifacts and SLAM map | `images/` plus `sparse/0/{cameras,images,points3D}.txt` | image references resolve; camera/image/point files parse |
| Splatfacto training | `bash scripts/train_splatfacto.sh` | validated COLMAP dataset and Nerfstudio env | checkpoints, config and training logs | final config/checkpoint reloads; no non-finite loss is reported |
| Gaussian export | `bash scripts/export_splat.sh [config.yml]` | trained Splatfacto config | Gaussian `.ply` | PLY exists, is non-empty and can be reopened by a compatible viewer |
| Camera-path render | `bash scripts/render_camera_path.sh PATH [config.yml]` | saved camera path and trained config | MP4 under `renders/` | FFprobe opens the video and reports a positive frame count |

## Runtime directories

Only source code and small configuration files belong in Git. The network volume holds datasets, source checkouts, environments, caches, logs, checkpoints, exports, and renders. This makes pod replacement safe as long as `/workspace` is mounted from the intended network volume.

## Version policy

- VIPE and Nerfstudio are pinned by 40-character Git commit SHA.
- VIPE installs from its upstream `uv.lock` using `uv sync --frozen`.
- Nerfstudio is pinned to release 1.1.5, but not to its documented torch 2.1.2/CUDA 11.8 stack: that only builds up to sm_90. torch 2.9.0+cu128, torchvision 0.24.0+cu128, Python 3.12 and NumPy 2.5.3 were raised together, keeping Nerfstudio's exact gsplat 1.4.0 pin.
- The RunPod image is pinned by a non-`latest` tag. `verify_environment.sh host` records the actual image/runtime facts after provisioning.
- Each setup script writes a package freeze and hardware report for the final submission.

## Failure policy

Scripts use strict shell settings and stop at the first failed check. Re-running setup is safe: existing Git checkouts must match their configured remote and are reset only by an explicit checkout of the pinned commit; output-producing stages refuse ambiguous or missing inputs. Long GPU stages write logs beneath `LOG_DIR`.
