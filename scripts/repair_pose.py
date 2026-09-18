#!/usr/bin/env python3
"""Repair one wrong relative camera pose using VIPE's own metric depth.

The relative pose is re-solved with RGB-D PnP: matched keypoints in the first
frame are back-projected with that frame's depth map, then the second camera is
located against those 3D points. The correction is composed into every later
pose so that all other relative poses are preserved.

The pose convention and the depth scale are not assumed. They are established
by reproducing a known-good pair and checking the prediction against the solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


def load_depth(archive: Path, index: int) -> np.ndarray:
    """Read one metric depth map.

    VIPE writes a single float16 "Z" channel, which OpenCV's EXR reader returns
    as zeros because it looks for colour channels, so read it with OpenEXR.
    """
    import OpenEXR

    with zipfile.ZipFile(archive) as handle:
        name = sorted(n for n in handle.namelist() if not n.endswith("/"))[index]
        with handle.open(name) as member:
            exr = OpenEXR.InputFile(member)
            window = exr.header()["dataWindow"]
            width = window.max.x - window.min.x + 1
            height = window.max.y - window.min.y + 1
            channel = exr.channels(["Z"])[0]

    return np.frombuffer(channel, dtype=np.float16).reshape(height, width).astype(np.float64)


def matched_points(first: Path, second: Path, features: int) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(features)
    left, right = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE), cv2.imread(str(second), cv2.IMREAD_GRAYSCALE)
    left_kp, left_desc = sift.detectAndCompute(left, None)
    right_kp, right_desc = sift.detectAndCompute(right, None)
    matches = cv2.BFMatcher().knnMatch(left_desc, right_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    source = np.float64([left_kp[m.queryIdx].pt for m in good])
    target = np.float64([right_kp[m.trainIdx].pt for m in good])
    return source, target


def solve_relative(
    first_image: Path,
    second_image: Path,
    depth: np.ndarray,
    camera: np.ndarray,
    features: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the 4x4 transform taking frame-1 camera coordinates into frame 2."""
    source, target = matched_points(first_image, second_image, features)
    if len(source) < 12:
        raise RuntimeError(f"only {len(source)} matches available")

    height, width = depth.shape
    columns = np.round(source[:, 0]).astype(int).clip(0, width - 1)
    rows = np.round(source[:, 1]).astype(int).clip(0, height - 1)
    sampled = depth[rows, columns]

    usable = np.isfinite(sampled) & (sampled > 0)
    source, target, sampled = source[usable], target[usable], sampled[usable]
    if len(source) < 12:
        raise RuntimeError(f"only {len(source)} matches carry valid depth")

    focal_x, focal_y = camera[0, 0], camera[1, 1]
    center_x, center_y = camera[0, 2], camera[1, 2]
    points = np.stack(
        [
            (source[:, 0] - center_x) / focal_x * sampled,
            (source[:, 1] - center_y) / focal_y * sampled,
            sampled,
        ],
        axis=1,
    )

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points,
        target,
        camera,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        iterationsCount=5000,
        confidence=0.9999,
    )
    if not ok or inliers is None or len(inliers) < 12:
        raise RuntimeError("PnP failed to find a supported pose")

    rvec, tvec = cv2.solvePnPRefineLM(
        points[inliers[:, 0]], target[inliers[:, 0]], camera, None, rvec, tvec
    )

    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()

    projected, _ = cv2.projectPoints(points[inliers[:, 0]], rvec, tvec, camera, None)
    residual = float(np.linalg.norm(projected.reshape(-1, 2) - target[inliers[:, 0]], axis=1).mean())

    return transform, {
        "matches": int(len(source)),
        "pnp_inliers": int(len(inliers)),
        "mean_reprojection_px": residual,
        "median_depth": float(np.median(sampled)),
    }


def predict(previous: np.ndarray, transform: np.ndarray, convention: str) -> np.ndarray:
    """Predict the next pose under a camera-to-world or world-to-camera reading."""
    if convention == "c2w":
        return previous @ np.linalg.inv(transform)
    return transform @ previous


def difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    relative = np.linalg.inv(left) @ right
    return rotation_angle(relative), float(np.linalg.norm(relative[:3, 3]))


def relative_pose(previous: np.ndarray, current: np.ndarray, convention: str) -> np.ndarray:
    if convention == "c2w":
        return np.linalg.inv(previous) @ current
    return current @ np.linalg.inv(previous)


def apply_correction(
    poses: np.ndarray, first_changed: int, corrected: np.ndarray, convention: str
) -> np.ndarray:
    """Set one pose and carry the same rigid correction into every later pose.

    Left-multiplication preserves camera-to-world relatives; world-to-camera
    poses need the correction on the right instead.
    """
    repaired = poses.copy()
    previous = poses[first_changed]
    if convention == "c2w":
        correction = corrected @ np.linalg.inv(previous)
        repaired[first_changed:] = correction @ poses[first_changed:]
    else:
        correction = np.linalg.inv(previous) @ corrected
        repaired[first_changed:] = poses[first_changed:] @ correction
    return repaired


def repair_slam_map(
    source: Path,
    destination: Path,
    original: np.ndarray,
    repaired: np.ndarray,
    first_changed: int,
    convention: str,
) -> dict[str, object]:
    """Move the points contributed by corrected frames by the same rigid transform.

    The map was triangulated against the original poses, so points seen from the
    shifted frames would otherwise stay behind their cameras.
    """
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    points = payload["dense_disp_xyz"].to(torch.float64).numpy()
    packs = payload["dense_disp_packinfo"].reshape(-1, 2).numpy()
    frames = list(payload["dense_disp_frame_inds"])

    if convention == "c2w":
        correction = repaired[first_changed] @ np.linalg.inv(original[first_changed])
    else:
        correction = np.linalg.inv(repaired[first_changed]) @ original[first_changed]

    moved = np.zeros(len(points), dtype=bool)
    for (offset, count), frame in zip(packs, frames):
        if int(frame) >= first_changed:
            moved[int(offset) : int(offset) + int(count)] = True

    homogeneous = np.concatenate([points[moved], np.ones((int(moved.sum()), 1))], axis=1)
    points[moved] = (homogeneous @ correction.T)[:, :3]
    payload["dense_disp_xyz"] = torch.from_numpy(points).to(torch.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "points_total": int(len(points)),
        "points_moved": int(moved.sum()),
        "packs_moved": int(sum(1 for frame in frames if int(frame) >= first_changed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--depth-zip", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=int, help="Index i of the wrong relative pose i -> i+1")
    parser.add_argument(
        "--validate-pair",
        type=int,
        required=True,
        help="Index of a pair the solver got right, used to establish convention and scale",
    )
    parser.add_argument("--features", type=int, default=8000)
    parser.add_argument("--max-rotation-error", type=float, default=3.0)
    parser.add_argument("--max-translation-error", type=float, default=0.25)
    parser.add_argument("--slam-map", type=Path, help="SLAM map whose points follow the corrected poses")
    parser.add_argument("--output-slam-map", type=Path)
    parser.add_argument("--output-poses", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    with np.load(args.poses) as handle:
        poses, indices = handle["data"].astype(np.float64), handle["inds"]
    intrinsics = np.load(args.intrinsics)["data"].astype(np.float64)
    frames = sorted(args.images.glob("*.jpg"))

    camera = np.array(
        [
            [intrinsics[0, 0], 0.0, intrinsics[0, 2]],
            [0.0, intrinsics[0, 1], intrinsics[0, 3]],
            [0.0, 0.0, 1.0],
        ]
    )

    # Establish the convention on a pair the solver is known to have got right.
    validation_transform, validation_stats = solve_relative(
        frames[args.validate_pair],
        frames[args.validate_pair + 1],
        load_depth(args.depth_zip, args.validate_pair),
        camera,
        args.features,
    )
    actual = poses[args.validate_pair + 1]
    candidates = {
        convention: difference(actual, predict(poses[args.validate_pair], validation_transform, convention))
        for convention in ("c2w", "w2c")
    }
    convention = min(candidates, key=lambda key: candidates[key][0])
    rotation_error, translation_error = candidates[convention]
    step = float(np.linalg.norm(poses[args.validate_pair + 1][:3, 3] - poses[args.validate_pair][:3, 3]))
    relative_translation_error = translation_error / step if step else float("inf")

    validation = {
        "pair": [args.validate_pair, args.validate_pair + 1],
        "convention": convention,
        "rotation_error_deg": rotation_error,
        "translation_error": translation_error,
        "translation_error_relative": relative_translation_error,
        "solver_step": step,
        **validation_stats,
        "candidates": {key: {"rotation_deg": value[0], "translation": value[1]} for key, value in candidates.items()},
    }
    print(json.dumps({"validation": validation}, indent=2))

    if rotation_error > args.max_rotation_error or relative_translation_error > args.max_translation_error:
        print(
            "Refusing to repair: the procedure does not reproduce a known-good pair"
            f" (rotation {rotation_error:.2f} deg, translation {relative_translation_error:.1%})",
            file=sys.stderr,
        )
        return 1

    transform, stats = solve_relative(
        frames[args.pair],
        frames[args.pair + 1],
        load_depth(args.depth_zip, args.pair),
        camera,
        args.features,
    )
    corrected = predict(poses[args.pair], transform, convention)
    repaired = apply_correction(poses, args.pair + 1, corrected, convention)

    old_relative = rotation_angle(relative_pose(poses[args.pair], poses[args.pair + 1], convention))
    new_relative = rotation_angle(relative_pose(repaired[args.pair], repaired[args.pair + 1], convention))

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_poses": str(args.poses),
        "validation": validation,
        "repair": {
            "pair": [args.pair, args.pair + 1],
            **stats,
            "rotation_before_deg": old_relative,
            "rotation_after_deg": new_relative,
            "step_before": float(np.linalg.norm(poses[args.pair + 1][:3, 3] - poses[args.pair][:3, 3])),
            "step_after": float(np.linalg.norm(repaired[args.pair + 1][:3, 3] - repaired[args.pair][:3, 3])),
            "poses_shifted": int(len(poses) - args.pair - 1),
        },
    }
    print(json.dumps(document["repair"], indent=2))

    if args.slam_map is not None and args.output_slam_map is not None:
        document["slam_map"] = repair_slam_map(
            args.slam_map, args.output_slam_map, poses, repaired, args.pair + 1, convention
        )
        print(json.dumps(document["slam_map"], indent=2))

    if args.output_poses is not None:
        args.output_poses.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_poses, data=repaired.astype(np.float32), inds=indices)
        print(f"wrote {args.output_poses}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
