#!/usr/bin/env python3
"""Validate a VIPE run: artifact groups, camera poses, intrinsics, trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED_GROUPS = ("pose", "intrinsics", "depth", "mask", "rgb")
ORTHONORMAL_TOLERANCE = 1e-3
# Tracking jumps are found with a robust outlier test: a step this many scaled
# MADs above the median is a jump rather than ordinary camera motion. A plain
# multiple of the median is too lax on real tracks, where the median step is
# already large; it is only the fallback for a perfectly uniform track.
CONTINUITY_SIGMAS = 10.0
CONTINUITY_FACTOR = 8.0


def load_indexed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as handle:
        return handle["data"], handle["inds"]


def check_poses(data: np.ndarray, problems: list[str]) -> dict[str, object]:
    if data.ndim != 3 or data.shape[1:] != (4, 4):
        problems.append(f"pose array has shape {data.shape}, expected (N, 4, 4)")
        return {}
    if not np.isfinite(data).all():
        problems.append("pose array contains non-finite values")
        return {}

    poses = data.astype(np.float64)
    rotations = poses[:, :3, :3]
    identity = np.eye(3)
    orthonormal_error = float(
        np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - identity).max()
    )
    if orthonormal_error > ORTHONORMAL_TOLERANCE:
        problems.append(f"rotation blocks are not orthonormal (max error {orthonormal_error:.2e})")

    determinants = np.linalg.det(rotations)
    if np.abs(determinants - 1.0).max() > ORTHONORMAL_TOLERANCE:
        problems.append(f"rotation determinants deviate from 1 (max {np.abs(determinants - 1.0).max():.2e})")

    bottom = poses[:, 3, :]
    if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        problems.append("pose matrices do not end with a [0, 0, 0, 1] row")

    centers = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(0)
    extent = centers.max(axis=0) - centers.min(axis=0)

    if len(steps) and float(steps.max()) == 0.0:
        problems.append("camera never moves between frames")

    jumps: list[dict[str, float]] = []
    median_step = float(np.median(steps)) if len(steps) else 0.0
    threshold = 0.0
    if median_step > 0:
        scaled_mad = 1.4826 * float(np.median(np.abs(steps - median_step)))
        threshold = (
            median_step + CONTINUITY_SIGMAS * scaled_mad
            if scaled_mad > 0
            else CONTINUITY_FACTOR * median_step
        )
        jumps = [
            {"from_frame": int(index), "to_frame": int(index) + 1, "step": float(step)}
            for index, step in enumerate(steps)
            if step > threshold
        ]
        if jumps:
            problems.append(
                f"{len(jumps)} discontinuous camera step(s) above {threshold:.3f}"
                f" (median {median_step:.3f})"
            )

    return {
        "orthonormal_max_error": orthonormal_error,
        "translation_extent": [float(value) for value in extent],
        "path_length": float(steps.sum()),
        "step_min": float(steps.min()) if len(steps) else None,
        "step_max": float(steps.max()) if len(steps) else None,
        "step_median": median_step,
        "continuity_threshold": threshold,
        "discontinuities": jumps,
        "centers": centers,
    }


def check_intrinsics(data: np.ndarray, problems: list[str]) -> dict[str, object]:
    if data.ndim != 2 or data.shape[1] != 4:
        problems.append(f"intrinsics array has shape {data.shape}, expected (N, 4)")
        return {}
    if not np.isfinite(data).all():
        problems.append("intrinsics array contains non-finite values")
        return {}
    if (data[:, :2] <= 0).any():
        problems.append("intrinsics contain a non-positive focal length")
    if (data[:, 2:] <= 0).any():
        problems.append("intrinsics contain a non-positive principal point")

    return {
        "fx_range": [float(data[:, 0].min()), float(data[:, 0].max())],
        "fy_range": [float(data[:, 1].min()), float(data[:, 1].max())],
        "principal_point": [float(data[0, 2]), float(data[0, 3])],
        "implied_resolution": [float(data[0, 2] * 2), float(data[0, 3] * 2)],
    }


def check_alignment(slam_map: Path, centers: np.ndarray, problems: list[str]) -> dict[str, object]:
    """Compare the SLAM point cloud with the camera track they were solved together from."""
    import torch  # imported lazily: only this check needs the VIPE environment

    payload = torch.load(slam_map, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "dense_disp_xyz" not in payload:
        problems.append("SLAM map has no dense_disp_xyz point cloud")
        return {}

    points = payload["dense_disp_xyz"].to(torch.float64).numpy()
    if not np.isfinite(points).all():
        problems.append("SLAM point cloud contains non-finite coordinates")
        return {"point_count": int(len(points))}

    lower, upper = points.min(axis=0), points.max(axis=0)
    cloud_extent = upper - lower
    margin = 0.5 * cloud_extent
    inside = np.all((centers >= lower - margin) & (centers <= upper + margin), axis=1)
    inside_fraction = float(inside.mean())
    if inside_fraction < 1.0:
        problems.append(
            f"{int((~inside).sum())} camera center(s) fall outside the point cloud bounds plus 50% margin"
        )

    scale = float(np.linalg.norm(cloud_extent))
    centroid_gap = float(np.linalg.norm(points.mean(axis=0) - centers.mean(axis=0)))
    if scale > 0 and centroid_gap / scale > 1.0:
        problems.append(f"camera and point cloud centroids are {centroid_gap / scale:.2f} cloud-diagonals apart")

    return {
        "point_count": int(len(points)),
        "cloud_extent": [float(value) for value in cloud_extent],
        "cameras_inside_fraction": inside_fraction,
        "centroid_gap": centroid_gap,
        "centroid_gap_over_cloud_diagonal": centroid_gap / scale if scale > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="VIPE output directory")
    parser.add_argument("--sequence", default="zavod70", help="Artifact name used by the run")
    parser.add_argument("--expected-frames", type=int, help="Frame count the run should contain")
    parser.add_argument("--report", type=Path, help="Write the JSON summary here")
    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help="Load the SLAM point cloud and compare it with the camera track (needs torch)",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        parser.error(f"output directory not found: {args.output_dir}")

    problems: list[str] = []
    groups: dict[str, list[str]] = {}
    for group in REQUIRED_GROUPS:
        directory = args.output_dir / group
        files = sorted(item.name for item in directory.glob(f"{args.sequence}*")) if directory.is_dir() else []
        groups[group] = files
        if not files:
            problems.append(f"missing artifact group: {group}")

    slam_map = args.output_dir / "vipe" / f"{args.sequence}_slam_map.pt"
    if not slam_map.is_file():
        problems.append(f"missing SLAM map: {slam_map}")

    pose_path = args.output_dir / "pose" / f"{args.sequence}.npz"
    intrinsics_path = args.output_dir / "intrinsics" / f"{args.sequence}.npz"
    for path in (pose_path, intrinsics_path):
        if not path.is_file():
            problems.append(f"missing required file: {path}")
    if problems:
        print("\n".join(f"PROBLEM: {problem}" for problem in problems), file=sys.stderr)
        return 1

    poses, pose_indices = load_indexed(pose_path)
    intrinsics, intrinsic_indices = load_indexed(intrinsics_path)

    frame_count = len(poses)
    if len(intrinsics) != frame_count:
        problems.append(f"{frame_count} poses but {len(intrinsics)} intrinsics")
    if args.expected_frames and frame_count != args.expected_frames:
        problems.append(f"expected {args.expected_frames} frames, found {frame_count}")
    if not np.array_equal(pose_indices, intrinsic_indices):
        problems.append("pose and intrinsics frame indices differ")
    if not np.array_equal(pose_indices, np.arange(frame_count)):
        problems.append("frame indices are not a contiguous 0..N-1 range")

    pose_summary = check_poses(poses, problems)
    centers = pose_summary.pop("centers", None)
    intrinsics_summary = check_intrinsics(intrinsics, problems)

    alignment: dict[str, object] = {}
    if args.check_alignment and centers is not None:
        alignment = check_alignment(slam_map, centers, problems)

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(args.output_dir),
        "sequence": args.sequence,
        "frame_count": frame_count,
        "artifact_groups": groups,
        "slam_map_bytes": slam_map.stat().st_size if slam_map.is_file() else None,
        "poses": pose_summary,
        "intrinsics": intrinsics_summary,
        "alignment": alignment,
        "problems": problems,
    }

    print(json.dumps(document, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2), encoding="utf-8")

    if problems:
        print(f"FAILED with {len(problems)} problem(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
