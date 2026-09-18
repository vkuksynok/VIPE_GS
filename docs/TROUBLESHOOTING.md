# Defects hit while building this pipeline

Each entry records the symptom, how it surfaced, what was tried, and — where a fix failed — how that
was detected. The paths mentioned here are produced by a run on the volume; none of them are part of
this repository.

One principle did most of the work: **a solver cannot validate itself.** Almost every defect below was
caught by an external measurement — an independent estimate from the images, a checksum, arithmetic on
a file size — rather than by trusting what a tool reported about its own output.

---

## 1. Geometry

### 1.1 A wrong relative pose between frames 22 and 23

The hardest defect in the project. One misconception is worth clearing first.

> **The camera did not turn sharply there.** That was the point: the sharp rotation existed in ViPE's
> solve, not in the capture. The images show a **4.64°** rotation; ViPE reported **64.74°**. The
> solver invented motion that never happened.

#### Symptom

After the full run on 126 frames, the camera track contained exactly one anomalous jump:

| Quantity | Transition 22→23 | Rest of the sequence |
| --- | --- | --- |
| Camera step | 36.47 | median 4.89, 99th percentile 6.86 |
| Rotation | 64.74° | median ~5°, next largest 26.7° |

The second largest step anywhere in the sequence was 6.94. The anomaly was **5.3× larger** than any
other value.

#### How it surfaced, and how it nearly did not

The continuity check **missed it**. The threshold was "8 × median" = 39.1, and the jump was 36.47 —
just under. The check returned "no problems".

It was found by looking at the distribution rather than the verdict: 99th percentile 6.86, second
largest step 6.94, and one value at 36.47. A gap that large between "everything else" and a single
value is not natural.

The threshold was then replaced with a robust rule based on the median absolute deviation:
`median + 10 × 1.4826 × MAD`. On this data that gives ~14.6 instead of 39.1, so the defect is now
caught automatically. A test reproduces exactly this situation.

#### Four hypotheses, all ruled out by measurement

Rather than immediately turning configuration knobs, each possible explanation was tested separately.

**Hypothesis 1 — a gap in capture time.** A multi-second gap between frames 22 and 23 would make a
large displacement legitimate.
*Test:* the capture timestamps of all 125 intervals.
*Result:* every interval was **exactly 1.0 s**, without exception. Ruled out.

**Hypothesis 2 — insufficient overlap between the frames.** If tracking breaks on a sharp viewpoint
change, the first thing to check is whether the correspondences exist at all.
*Test:* ORB matching with RANSAC on pairs 19→20 through 26→27.
*Result:* the problem pair had 645 good matches and 208 RANSAC inliers — **no worse than its
neighbours** (330, 153, 143). There was ample material to solve with. Ruled out.

**Hypothesis 3 — the motion was genuinely sharp.** The decisive test.
*Test:* recover the relative rotation directly from the images with SIFT and an essential matrix, with
no involvement from ViPE.

| Pair | From the images | ViPE | Disagreement |
| --- | --- | --- | --- |
| 20→21 | 0.54° | 0.53° | 0.01° |
| 21→22 | 6.76° | 6.57° | 0.19° |
| **22→23** | **4.64°** | **64.74°** | **60.1°** |
| 23→24 | 3.92° | 2.96° | 0.96° |
| 24→25 | 8.87° | 8.72° | 0.15° |

On the neighbouring pairs the same method agrees with ViPE to within a degree — which proves the
method works and the indexing is not shifted. Only one pair disagrees, by 60°. This is a solver error,
not a property of the data. Ruled out.

**Hypothesis 4 — an MP4 encoding artifact.** ViPE reads the MP4 while the check reads the source
JPEGs. If the encoder damaged those specific frames, ViPE would be seeing something different.
*Test:* PSNR of each decoded MP4 frame against its source JPEG.
*Result:* a uniform **39.5–40.5 dB across all 126 frames**, with no dip at 22–25. The worst frames
(17–21, 39.5 dB) are barely below the median. Ruled out.

Throughout all of this, ViPE logged **no warning whatsoever** — it silently produced a wrong pose.

#### The tool that made the rest possible

It became clear that ranking runs by eye was not going to work. Hence
`scripts/crosscheck_pose_rotations.py`: it recovers the relative rotation of **every** consecutive
pair straight from the images and compares it with the solve, across 12 processes.

That tool is what showed which repair attempts actually worked and which only appeared to.

#### Three attempted fixes, and how each was found wanting

**Attempt 1 — a robust kernel, `robust_kernel=huber`.**
*Rationale:* the SLAM backend uses an L2 loss, which is sensitive to outliers in the flow residuals. A
robust kernel should suppress the outlier.
*It looked like a success locally:* that specific defect disappeared — step 36.47 → 3.69, rotation
64.74° → 2.54°, close to the true 4.64°.
*How the failure was detected:* the cross-check across the whole sequence. Instead of one jump there
were now **three** (35.49 / 33.52 / 32.74), the largest rotation grew to 107.8°, and disagreeing pairs
went from one to **56 of 122**. Median error rose from 0.26° to 4.28°.

This is the most instructive episode in the project. Had the inspection stopped at the problem pair,
the fix would have been accepted. A local improvement was hiding a global collapse.

**Attempt 2 — a different depth model, `pipeline=no_vda`.**
*Rationale:* after the huber result it was clear the outcome is sensitive to depth residuals, which
implicates the depth prior.
*Result:* the defect **did not disappear, it moved** — from pair [21, 22] to [23, 24], and grew:
74.1° instead of 60.8°. Overall accuracy was unchanged (median 0.28°, one disagreeing pair).
*Conclusion:* the depth model is not the cause.

**Attempt 3 — SLAM frontend thresholds, `frontend_thresh` 16→32 and `backend_thresh` 22→44.**
*Rationale:* the frontend adds edges between frames whose flow "distance" is below the threshold. At
1 fps the flow is large, so frames in the fast section could be left almost without edges, and
therefore under-constrained.
*Result:* it **moved** again, now to pair [22, 23] with a 72.6° error.

#### What that established

Four configurations produced the same picture: exactly one wrong relative pose, always within frames
22–25, only changing position.

| Run | Usable pairs | Median | 95th pct | Disagreeing > 5° | Worst |
| --- | --- | --- | --- | --- | --- |
| L2, default | 122 | 0.26° | 1.64° | **1** | [21,22]: 64.7° vs 4.0° |
| `no_vda` | 122 | 0.28° | 1.66° | **1** | [23,24]: 82.9° vs 8.9° |
| Raised SLAM thresholds | 122 | 0.27° | 1.72° | **1** | [22,23]: 75.1° vs 2.6° |
| `robust_kernel=huber` | 122 | 4.28° | 63.75° | **56** | [88,89]: 107.8° vs 16.1° |

The defect is robust to the depth model, to the SLAM thresholds, and is not an encoding artifact. The
only thing that distinguishes that stretch is speed of motion: ORB inliers fall from ~520 on pair
19→20 to ~70 on pair 26→27. At 1 fps the motion there exceeds what a flow-based tracker handles
reliably, and the solver settles into a wrong local minimum at one of the transitions. This looks like
the limit of ViPE on this kind of data rather than a matter of configuration.

An important counterpoint: **outside that single pair ViPE is very accurate here** — 121 of 122 pairs
agree with the images to within ~1.6°.

#### How it was finally fixed

Since configuration would not remove it, the wrong relative pose was re-solved from ViPE's own
artifacts by `scripts/repair_pose.py`:

1. SIFT correspondences between frames 22 and 23;
2. back-projection of the frame-22 points using that frame's **metric depth**, which ViPE already
   computed;
3. RANSAC PnP for frame 23, giving a relative pose in the same scale;
4. the rigid correction is composed into all 104 later poses, so every other relative pose is
   preserved.

**A guard against silent failure.** The pose convention (camera-to-world versus world-to-camera) and
the depth scale are not guessed. The script first reproduces a pair the solver is known to have got
right and compares its prediction against ViPE's. If it cannot reproduce that pair, it refuses to
change anything. The convention resolved unambiguously from the data:

| Candidate | Rotation error | Translation error |
| --- | --- | --- |
| `c2w` | 1.82° | 0.68 (17.9 % of the 3.83 step) |
| `w2c` | 2.69° | 8.32 |

| Repair of pair (21, 22) | Before | After |
| --- | --- | --- |
| Rotation | 64.74° | 7.33° |
| Step | 36.47 | 5.86 |
| PnP inliers / reprojection | — | 305 / 1.14 px |

**Verification of the repair:**

| Measure | Before | After |
| --- | --- | --- |
| Pairs disagreeing > 5° | 1 | **0** |
| Worst error | 60.79° | **3.37°** |
| Median / 95th percentile | 0.26° / 1.64° | **0.26° / 1.64°** — unchanged |
| Trajectory discontinuities | 1 | **0** |

The unchanged median and 95th percentile are the key evidence: the repair touched the defect rather
than smoothing the whole trajectory.

#### A consequence that nearly corrupted the next stage

The SLAM point cloud was triangulated against the **original** poses. After 104 frames shifted, the
points contributed by those frames would have been left behind their cameras — and would have gone
into COLMAP in that state as the initial 3D points for training.

The map's structure made a clean fix possible: `dense_disp_packinfo` and `dense_disp_frame_inds` map
every point to its source frame. The same rigid transform was applied to **262,270 of 308,812 points**
(102 of 120 packs). The alignment check afterwards: all 126 cameras inside the cloud bounds, centroid
gap 0.145 of the cloud diagonal.

#### Honest limits of this repair

- One of the 125 relative poses now comes from RGB-D PnP rather than from SLAM. The affected output
  directory carries a `REPAIRED.md` describing its provenance.
- The procedure's translation error on the control pair was 17.9 %, so the 5.86 step is accurate to
  roughly ±1.
- The robust MAD rule catches **local** jumps, not global degradation: on the huber run it reports
  zero outliers precisely because the whole trajectory is erratic there. That case is caught by the
  image cross-check, not by the continuity test.

### 1.2 Full-resolution stills are MPO, not JPEG

The first dataset check flagged all 126 frames of the full-resolution archive as an unexpected format.

*Cause:* DJI writes full-resolution stills as MPO (multi-picture) containers holding several images.
Pillow honestly reports `MPO`, and the check expected exactly `JPEG`.
*Why it is not a data problem:* MPO is JPEG-compatible, and Pillow, ffmpeg and COLMAP all read the
primary image without trouble.
*Fix:* the check accepts `MPO` alongside `JPEG` and records the format in its report. A false
positive, not a defect in the dataset.

### 1.3 The configured dataset path did not exist

`RAW_DATASET` pointed at a directory that was never created; the archives were somewhere else. Fixed
to the verified location, with `RAW_DATASET_FULL` and `RAW_FRAME_DIR` added.

### 1.4 What was sound about the dataset

So as not to leave the impression that the data was bad — everything else about it checked out:

- the SHA-256 of both archives matched the local copies;
- 126 frames each, numbered 1–126 with no gaps and no duplicates;
- every frame **fully decoded**, not merely header-checked;
- all 125 capture intervals exactly 1.0 s, which confirmed the 1 fps assumption by measurement rather
  than by guesswork;
- all 126 prepared frames traced back to the source archive by SHA-256, with zero unmatched and zero
  mismatched.

---

## 2. The ViPE environment

**The image ships no conda.** `setup_vipe.sh` needs it, and the RunPod PyTorch image does not have it.
`ensure_conda` installs a pinned Miniforge onto the network volume so the toolchain survives host
replacement.

**The CUDA extension failed on a Python header.** `vipe_ext` would not compile:
`fatal error: x86_64-linux-gnu/python3.12/pyconfig.h`. The trap is that `python3.12-dev` **is**
installed and the header does exist — but conda's compiler uses its own sysroot and does not search
`/usr/include`. Rather than mixing system headers into a conda toolchain build, which risks ABI
mismatches, the sync was pointed at a uv-managed interpreter whose headers are self-contained. The
extension then built for the local architecture.

**Model weights landed on ephemeral storage.** The first run downloaded 1.5 GB of Video-Depth-Anything
weights into the container disk, which is wiped when the pod stops. `TORCH_HOME` now points at the
volume; a warm run downloaded nothing.

**ViPE saves no resolved config.** Its `default.yaml` sets `hydra.output_subdir: null`, so no record of
the run's configuration remains. An explicit dump was added. Note that `--cfg job --resolve` cannot be
used: ViPE registers its own `neq` interpolation resolver at runtime and Hydra fails with
`UnsupportedInterpolationType`. The dump therefore uses `--cfg job`.

**`VIPE_SEQUENCE` does not name the artifacts.** ViPE takes the artifact name from the video filename,
not from the variable. A smoke run with `VIPE_SEQUENCE=zavod70-smoke` produced files named
`zavod70.*`, and the first attempt to locate its results found nothing. The variable only controls log
filenames.

---

## 3. The Nerfstudio environment and Blackwell

**The documented stack does not support this GPU.** torch 2.1.2 / CUDA 11.8 builds up to sm_90, and
the GPU reports `capability (12, 0)` — sm_120. The pins were raised **together**: Python 3.12,
CUDA 12.8, torch 2.9.0+cu128, torchvision 0.24.0+cu128, numpy 2.5.3. Python had to move because torch
2.9.0+cu128 publishes no cp310 wheel. Nerfstudio stayed at 1.1.5, which is still the latest release,
and gsplat at 1.4.0, because Nerfstudio pins it exactly and gsplat ships pure Python, compiling its
kernels for the local architecture.

**`conda create` silently discarded packages.** Specs placed after `-c` are rejected by the parser with
`unrecognized arguments`, so the environment was never created at all. Channels must come **before**
the packages.

**gsplat disabled itself quietly.** Calling `ns-train` directly does not activate the conda
environment, so `CUDA_HOME` and `nvcc` were absent. gsplat announced this in **a single log line** —
`gsplat: No CUDA toolkit found. gsplat will be disabled.` — and then failed far later with
`AttributeError: 'NoneType' object has no attribute 'CameraModelType'`. The error at the point of
failure had nothing to do with the cause. `use_nerfstudio_env` now sets the variables.

**CUDA headers are not where torch looks.** conda keeps them under `targets/<arch>-linux/include`,
while torch's extension builder searches `$CUDA_HOME/include`, so gsplat's kernels failed to compile on
`cuda_runtime_api.h`. `CPATH` and `LIBRARY_PATH` were added.

**Nerfstudio could not reload its own checkpoint.** torch 2.6 changed the default of `weights_only`
from `False` to `True`, and Nerfstudio 1.1.5 predates that and calls `torch.load` without the argument.
Training saved checkpoints fine, but `ns-eval` and `ns-export` failed with `UnpicklingError`.
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set; only checkpoints produced by this pipeline are ever
loaded.

**`ns-export` needs OpenGL libraries.** It imports open3d, which links against `libEGL`; the image has
neither `libEGL` nor `libGL`. `ensure_open3d_libraries` installs them.

**Run parameters could not be overridden.** `versions.env` promised in its own comment that run
parameters could be set from the environment, but assigned them unconditionally. A smoke run asking
for 2000 iterations **silently started 30000**. Caught in the first line of the log. The values now use
the `${VAR:-default}` form; the version pins deliberately remain hard.

**Training computed no metrics and had no held-out split.** `train_splatfacto.sh` had no eval split, so
renders could only be compared against frames the model had fitted. `--eval-mode interval
--eval-interval 8` was added. Separately, with the default `--vis viewer` Nerfstudio computes no
metrics — it says so in its own log line — so PSNR/SSIM are measured afterwards with `ns-eval`.

---

## 4. Infrastructure

**The SSH proxy stalls on non-interactive commands.** `ssh.runpod.io` answers
`Your SSH client doesn't support PTY`, and with a forced PTY it simply hangs, returning nothing — a
backgrounded attempt finished many minutes later with empty output. All work moved to the direct TCP
endpoint from the pod's runtime ports.

**The external port changes on every restart.** Confirmed three times. Read it again before each
connection.

**The container disk is discarded on stop.** Anything worth keeping — session logs in particular —
must be copied onto the volume first.

**`df` does not report what you expect.** For a network volume it returns the capacity of the
underlying MooseFS cluster, not the volume quota. Use `du`. A related subtlety: uv hardlinks its cache
into the virtual environment, so per-directory totals add up to more than the volume total; only the
total is meaningful.

**A stopped pod may not restart.** RunPod answers `There are not enough free GPUs on the host machine
to start this pod` — while a pod sits stopped, its host can hand the GPU to someone else. The volume
is unaffected: a new pod on a different host picked up every environment and artifact untouched, which
is exactly what putting them on the volume was for.

---

## 5. Defects in the verification tooling itself

A separate section, because several problems were in the checks rather than in the pipeline.

**The continuity threshold was too lax** — "8 × median" missed the main defect of the project (36.47
against a threshold of 39.1). Replaced with the robust MAD rule.

**The cross-check counted degenerate pairs as disagreements.** When the essential matrix degenerates,
`recoverPose` returns a meaningless ~180° with **zero** inliers. Those were initially scored as solver
errors, which made `no_vda` look worse than it was. `--min-inliers` now treats too few inliers as
unusable evidence rather than as a disagreement.

**OpenCV silently returned zeros instead of depth.** ViPE writes depth to EXR as a single `Z` channel
in float16. `cv2.imdecode` looks for colour channels and returns an array of the correct shape
**filled with zeros**, without raising anything. It surfaced as "0 of 1961 matches carry valid depth".
The reader was switched to OpenEXR, which is how ViPE reads it too.

**The model size was misread.** A number taken from the training log's `Train Rays / Sec` column was
reported as the gaussian count. The real figure is 7,239,301, of which 7,055,547 were exported.

**Python wrote CRLF into shell scripts.** Patching files from Python on Windows broke `bash` with
`syntax error near unexpected token`. Files were normalised back to LF, and `.gitattributes` enforces
`eol=lf`.

---

## 6. What remains unresolved

**Reconstruction quality is modest.** PSNR 15.13 on held-out frames against 22.29 on training frames.
The ~7 dB gap means the model reproduces what it saw and generalises weakly. Likely causes in order:
scene content (leafless vegetation is high-frequency and view-dependent), the sparse-view regime, large
baselines at 1 fps, and residual pose error whose 95th percentile is 1.64° — tens of pixels at
`fx ≈ 1435`.

The most promising untried lever is Splatfacto's camera optimizer
(`--pipeline.model.camera-optimizer.mode SO3xR3`), which refines poses during training and targets that
last cause directly. Per-image exposure embeddings and depth supervision from ViPE's metric depth are
the next two.

**Untried options for the tracking defect:** the `dav3` depth preprocessor, and `frame_dir_stream`
instead of an MP4. After the encoding artifact was ruled out by the PSNR measurement, the second holds
little promise.

**The repair was never A/B tested.** Training was only ever run on the repaired poses, so there is no
measurement of what the repair is worth in final quality. What it demonstrably prevented is a visible
lurch in the rendered fly-through, since the camera path is built from those same poses.
