# Verified environment

What the pipeline was actually built and measured on. This describes a reference run; the artifacts it
mentions are written to the network volume by running the pipeline and are not part of this
repository. Transient connection details — public IP, external port, SSH user, key material — are
deliberately omitted.

## Host

| Field | Verified value |
| --- | --- |
| Cloud and data center | RunPod Secure Cloud, `EU-RO-1` |
| Pod template and image | `runpod-torch-v280`; `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB VRAM, compute capability 12.0 |
| Host CUDA | 13.0 |
| NVIDIA driver | 580.173.02 |
| Python in the image | 3.12.3 |
| PyTorch in the image | 2.8.0+cu128, CUDA available |
| Persistent storage | network volume mounted at `/workspace` |

The image is a runtime image: it ships **no `nvcc` and no conda**, both of which the pipeline needs.
`setup_vipe.sh` installs a pinned Miniforge onto the volume and takes the CUDA toolchain from there.

## ViPE environment

Built by `scripts/setup_vipe.sh`.

| Field | Verified value |
| --- | --- |
| Conda | Miniforge 26.7.2-0, installed into `envs/miniforge` on the volume |
| CUDA toolkit | `nvcc` 12.8.61, from ViPE's own `envs/cu128.yml` |
| Python | 3.12.14, uv-managed |
| PyTorch | 2.9.0+cu128, CUDA available |
| ViPE commit | `8c9f36144e08d8f8c8cf60d20ad70c100d9cff07`, matching the pin |

Two obstacles were resolved rather than worked around:

- The image ships no conda, so `ensure_conda` installs a pinned Miniforge onto the volume.
- Building the `vipe_ext` CUDA extension failed on
  `#include <x86_64-linux-gnu/python3.12/pyconfig.h>`. The `python3.12-dev` package is present and the
  header exists, but conda's compiler uses its own sysroot and does not search `/usr/include`. Rather
  than mixing system headers into a conda toolchain build, the sync targets a uv-managed interpreter
  whose headers are self-contained.

Model weights must stay on the volume: ViPE pulls roughly 1.5 GB of Video-Depth-Anything weights
through `torch.hub`, which defaults to the container disk and is discarded when the pod stops.
`TORCH_HOME` points at the volume cache instead.

## Nerfstudio environment

Built by `scripts/setup_splatfacto.sh`. The stack Nerfstudio documents — torch 2.1.2 with CUDA 11.8 —
only builds up to sm_90, and this GPU is sm_120, so the pins were raised together.

| Field | Verified value |
| --- | --- |
| Python | 3.12 — torch 2.9.0+cu128 publishes no cp310 wheel |
| CUDA toolkit | 12.8 in the environment, `nvcc` 12.8.61 |
| PyTorch | 2.9.0+cu128, CUDA available, capability `(12, 0)` |
| numpy | 2.5.3 |
| gsplat | 1.4.0 — the exact pin Nerfstudio carries; pure Python, compiles kernels for the local architecture |
| Nerfstudio | 1.1.5, commit `6b60855003011b2ca23c2fe3f8e2ca6314c69924` |

Raising the pins was possible because Nerfstudio 1.1.5 only caps torch below 2.2 in its `dev` extra,
not at runtime.

Five image-specific defects had to be fixed first:

- `conda create` rejects every package spec placed after `-c`, so the environment was never created.
- Calling `ns-train` directly never activates the environment, leaving `CUDA_HOME` and `nvcc` absent;
  gsplat disabled itself in one log line and failed far later on a `None` extension.
  `use_nerfstudio_env` sets them.
- conda keeps CUDA headers under `targets/<arch>-linux/include`, not in `$CUDA_HOME/include` where
  torch's extension builder looks, so the kernels failed to compile on `cuda_runtime_api.h`.
- Nerfstudio 1.1.5 predates the torch 2.6 switch to `weights_only=True` and calls `torch.load` without
  it, so it could not reload its own checkpoints.
- `ns-export` imports open3d, which links against `libEGL` and `libGL`; neither is in the image, so
  `ensure_open3d_libraries` installs them.

## Dataset that was used

126 DJI stills of an abandoned industrial site, supplied as two archives — a full-resolution set at
4000×3000 and a reduced set at 1920×1440, with identical filenames. The pipeline ran on the reduced
set.

| Property | Reduced archive | Full archive |
| --- | --- | --- |
| Frames | 126 | 126 |
| Resolution | 1920×1440 | 4000×3000 |
| Pillow format | JPEG | MPO |
| Sequence numbers | 1–126, no gaps, no duplicates | 1–126, no gaps, no duplicates |

`testzip()` reported no corrupt members and every frame was fully decoded rather than only
header-checked. Capture timestamps span 125 seconds for 126 frames, which is where the 1 fps assumption
used downstream comes from — measured, not assumed.

The full-resolution stills are DJI multi-picture (MPO) containers. Pillow, ffmpeg and COLMAP all read
the primary image, so the verifier accepts MPO alongside JPEG.

## Working tree on the volume

A run creates `/workspace/gs_test/` holding `data/`, `third_party/`, `envs/`,
`cache/{huggingface,torch,uv,pip,torch_extensions}/`, `outputs/`, `artifacts/environment/`,
`exports/`, `renders/` and `logs/`. Source archives stay untouched wherever they were placed.

Budget roughly 45 GB: the two environments alone account for about 30 GB, mostly torch with the NVIDIA
CUDA libraries.

Measuring usage needs care. `df` reports the capacity of the underlying MooseFS cluster rather than the
volume quota, so use `du`. And because uv hardlinks its cache into the virtual environment,
per-directory totals add up to more than the volume total — only the total is meaningful.

## Operational notes

- The pod's external TCP port is reassigned on every start. Read the current details from the pod
  runtime before each connection.
- Connect over the direct TCP endpoint, not the `ssh.runpod.io` proxy: the proxy requires a PTY and
  stalls on non-interactive command execution, which is how these scripts are driven.
- The container disk is discarded when the pod stops. Copy anything worth keeping — session logs in
  particular — onto the volume first.
- A stopped pod may refuse to start again with *not enough free GPUs on the host machine*: while it
  sits stopped, its host can hand the GPU to another tenant. The volume is unaffected, and a new pod
  picks the environments up untouched.
