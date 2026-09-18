# Runbook

Every command in order, with timings from the reference run: an NVIDIA RTX PRO 4500 Blackwell (32 GB),
126 stills at 1920×1440, dataset and environments on a network volume.

Paths below assume the defaults from `config/paths.env.example`: the working tree at
`/workspace/gs_test` and the source archives under `/workspace/datasets`.

## 0. Host

- Linux, NVIDIA GPU with at least 24 GB — ViPE peaked at 24,806 MiB on 126 frames at 1920×1440
- Persistent storage mounted at `/workspace`; on RunPod the pod and the volume must be in the **same
  data center** or the volume will not mount
- On RunPod: connect over the **direct** TCP endpoint from the pod's runtime ports. The
  `ssh.runpod.io` proxy requires a PTY and stalls on non-interactive commands, which is how these
  scripts are driven. The external port is reassigned every time the pod starts.

## 1. Repository

```bash
cd /workspace/gs_test
git clone <this repo> repo
cd repo && chmod +x scripts/*.sh
```

## 2. Host check — 10 s

```bash
bash scripts/verify_environment.sh host
```

Reports the GPU, CUDA and that `/workspace` is mounted and writable.

## 3. Environments — 25 min + 30 min, once per volume

```bash
bash scripts/setup_vipe.sh
bash scripts/setup_splatfacto.sh
```

Both install onto the volume and survive host replacement, so skip this on a volume that already has
them. `setup_vipe.sh` installs Miniforge first: the image ships no conda, and ViPE's `envs/cu128.yml`
needs it for `nvcc`, eigen and the cuBLAS/cuSOLVER development packages.

## 4. Dataset — 1 min

```bash
bash scripts/verify_dataset.sh
```

Whole-archive SHA-256, ZIP integrity, DJI filename parsing, duplicate and sequence-gap detection, a
full decode of every frame, and a checksum-verified extraction. Writes a JSON report under
`artifacts/environment/`.

## 5. Prepare the sequence — 2 min

```bash
python3 scripts/prepare_dataset.py \
  --input /workspace/datasets/Zavod70_small.zip \
  --output /workspace/gs_test/data/prepared/zavod70 \
  --fps 1 --max-width 1920

REPORT=$(ls -t /workspace/gs_test/artifacts/environment/dataset-*.json | head -1)
python3 scripts/verify_prepared_sequence.py \
  --prepared-dir /workspace/gs_test/data/prepared/zavod70 \
  --expected-count 126 --dataset-report "$REPORT"
```

The manifest must report all frames encoded, and the check must trace every prepared frame back to the
source archive by SHA-256.

## 6. ViPE — 6 min

```bash
bash scripts/run_vipe.sh
```

Writes poses, intrinsics, metric depth, masks, the SLAM map and a visualisation video, plus the
resolved Hydra config and the run log.

## 7. Verify the solve — 1.5 min

These scripts import `torch`, `cv2` and `numpy`, so they run inside the ViPE environment. Define the
wrapper once:

```bash
VP() { UV_PROJECT_ENVIRONMENT=/workspace/gs_test/envs/vipe-python \
  /workspace/gs_test/envs/miniforge/bin/conda run --prefix /workspace/gs_test/envs/vipe-cu128 \
  uv run --project /workspace/gs_test/third_party/vipe python "$@"; }
```

```bash
VP scripts/verify_vipe_outputs.py \
  --output-dir /workspace/gs_test/outputs/vipe --sequence zavod70 \
  --expected-frames 126 --check-alignment

VP scripts/crosscheck_pose_rotations.py \
  --images /workspace/gs_test/data/prepared/zavod70/images \
  --poses /workspace/gs_test/outputs/vipe/pose/zavod70.npz \
  --intrinsics /workspace/gs_test/outputs/vipe/intrinsics/zavod70.npz \
  --workers 12
```

**Do this before training.** The first script checks the solve against itself — artifact groups,
finite poses, orthonormal rotations, trajectory continuity, cameras inside the point cloud. The second
checks it against the images, which is the only way a solver error becomes visible: it recovers each
consecutive rotation with SIFT and an essential matrix and compares.

On the reference dataset it reports one disagreeing pair, `[21, 22]`, at 60.79°.

## 8. Repair a bad pose — 1 min, conditional

Only if the cross-check found a disagreement. Otherwise skip this and keep using `outputs/vipe` below.

```bash
VP scripts/repair_pose.py --pair 21 --validate-pair 10 \
  --poses /workspace/gs_test/outputs/vipe/pose/zavod70.npz \
  --intrinsics /workspace/gs_test/outputs/vipe/intrinsics/zavod70.npz \
  --depth-zip /workspace/gs_test/outputs/vipe/depth/zavod70.zip \
  --images /workspace/gs_test/data/prepared/zavod70/images \
  --slam-map /workspace/gs_test/outputs/vipe/vipe/zavod70_slam_map.pt \
  --output-slam-map /workspace/gs_test/outputs/vipe-repaired/vipe/zavod70_slam_map.pt \
  --output-poses /workspace/gs_test/outputs/vipe-repaired/pose/zavod70.npz

cd /workspace/gs_test/outputs/vipe-repaired
for d in depth mask rgb intrinsics; do [ -e $d ] || ln -s ../vipe/$d $d; done
cd /workspace/gs_test/repo
```

`--pair` is the index reported by the cross-check; `--validate-pair` must be a pair the solver got
right. The symlinks matter: the COLMAP converter expects a complete artifact set, and only the poses
and the SLAM map were rewritten.

Re-run the cross-check against `outputs/vipe-repaired` to confirm zero disagreeing pairs.

## 9. COLMAP dataset — 2 min

```bash
VIPE_OUTPUT_DIR=/workspace/gs_test/outputs/vipe-repaired bash scripts/prepare_colmap.sh
python3 scripts/verify_colmap_dataset.py --dataset /workspace/gs_test/data/splatfacto/zavod70
```

Drop the `VIPE_OUTPUT_DIR` override if no repair was needed. Expect `"problems": []`, one `PINHOLE`
camera, and every image reference resolving to a file. The check also recovers the camera track from
`images.txt` as `-Rᵀt`, which is how you confirm the repaired poses actually reached the dataset
rather than being lost in conversion.

## 10. Training — 44 min

```bash
bash scripts/train_splatfacto.sh
```

30,000 iterations, seed 42, every 8th frame held out of optimization. On a volume where gsplat has not
run before, add about 8 minutes: it compiles its CUDA kernels for the local architecture on first use,
then caches them in `TORCH_EXTENSIONS_DIR` on the volume.

`ns-train` prints a viewer URL. On a headless host it goes nowhere; that is expected.

## 11. Evaluate, export, render — 20 min

```bash
bash scripts/eval_splatfacto.sh     # 9 min  → PSNR/SSIM/LPIPS on the held-out frames
bash scripts/export_splat.sh        # 5 min  → splat.ply

CFG=$(ls -t /workspace/gs_test/outputs/splatfacto/*/splatfacto/*/config.yml | head -1)
CUDA_HOME=/workspace/gs_test/envs/nerfstudio \
PATH=/workspace/gs_test/envs/nerfstudio/bin:$PATH \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/make_camera_path.py --config "$CFG" \
  --output /workspace/gs_test/artifacts/camera_path.json

bash scripts/render_camera_path.sh /workspace/gs_test/artifacts/camera_path.json   # 3.5 min
```

`make_camera_path.py` takes the trajectory from the trained run's own dataparser, so the poses are
already in the model's world space and no convention is converted by hand. It interpolates the
captured cameras five-fold, which means most rendered frames are viewpoints that were never captured.

## 12. Demo video — no GPU needed

```bash
python scripts/make_demo.py \
  --frames <source zip or frame directory> \
  --flythrough <rendered camera-path mp4> \
  --comparison <one eval side-by-side png> \
  --output demo.mp4 --crf 26
```

Runs anywhere with ffmpeg and Pillow.

## Short run

Replace steps 5 and 10 to compress the pipeline to roughly 20 minutes:

```bash
python3 scripts/prepare_dataset.py --input /workspace/datasets/Zavod70_small.zip \
  --output /workspace/gs_test/data/prepared/zavod70-smoke \
  --limit 20 --fps 1 --max-width 1280

INPUT_VIDEO=/workspace/gs_test/data/prepared/zavod70-smoke/zavod70.mp4 \
VIPE_OUTPUT_DIR=/workspace/gs_test/outputs/vipe-smoke \
VIPE_SEQUENCE=zavod70-smoke bash scripts/run_vipe.sh

SPLATFACTO_MAX_ITERATIONS=2000 bash scripts/train_splatfacto.sh
```

A 20-frame subset stops short of the problematic pair, so the pose defect will not appear in a smoke
run.

## Things that will bite

| Symptom | Cause |
| --- | --- |
| Pod will not start: *not enough free GPUs on the host machine* | A stopped pod's host can hand its GPU to another tenant. Create a new pod; the volume is unaffected. |
| SSH hangs with no output | The `ssh.runpod.io` proxy. Use the direct TCP endpoint. |
| `gsplat: No CUDA toolkit found`, then a `NoneType` error minutes later | A Nerfstudio entry point was called directly instead of through `scripts/*.sh`, so `CUDA_HOME` and `nvcc` were never set. |
| A stage exits with code −9 and no traceback | The OOM killer. ViPE caches the whole sequence in host RAM several times over; lower `--max-width`. |
| `du` and `df` disagree about volume usage | `df` reports the MooseFS cluster capacity, not the volume quota. Measure with `du`. |
| Session logs disappear after a stop | The container disk is discarded. Copy anything worth keeping onto `/workspace` first. |
