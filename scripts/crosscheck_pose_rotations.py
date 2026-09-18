#!/usr/bin/env python3
"""Cross-check solved camera rotations against image-derived relative rotations.

The solver's own poses cannot validate themselves. This recovers the relative
rotation of consecutive frames straight from the images with SIFT plus an
essential matrix, and reports where the solved poses disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def rotation_angle(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))))


def image_rotation(task: tuple[str, str, float, float, float, int, float]) -> tuple[int, float, int]:
    """Relative rotation between two images, in degrees, with its inlier count."""
    first, second, focal, cx, cy, index, scale = task
    flags = cv2.IMREAD_GRAYSCALE
    left, right = cv2.imread(first, flags), cv2.imread(second, flags)
    if scale != 1.0:
        left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    matrix = np.array([[focal * scale, 0.0, cx * scale], [0.0, focal * scale, cy * scale], [0.0, 0.0, 1.0]])
    sift = cv2.SIFT_create(4000)
    left_kp, left_desc = sift.detectAndCompute(left, None)
    right_kp, right_desc = sift.detectAndCompute(right, None)
    if left_desc is None or right_desc is None or len(left_kp) < 8 or len(right_kp) < 8:
        return index, float("nan"), 0

    matches = cv2.BFMatcher().knnMatch(left_desc, right_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return index, float("nan"), len(good)

    source = np.float32([left_kp[m.queryIdx].pt for m in good])
    target = np.float32([right_kp[m.trainIdx].pt for m in good])
    essential, mask = cv2.findEssentialMat(source, target, matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None or essential.shape != (3, 3):
        return index, float("nan"), int(mask.sum()) if mask is not None else 0

    _, rotation, _, pose_mask = cv2.recoverPose(essential, source, target, matrix, mask=mask)
    return index, rotation_angle(rotation), int(pose_mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory of ordered prepared frames")
    parser.add_argument("--poses", required=True, type=Path, help="VIPE pose .npz")
    parser.add_argument("--intrinsics", required=True, type=Path, help="VIPE intrinsics .npz")
    parser.add_argument("--scale", type=float, default=0.5, help="Downscale factor for matching")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Degrees of disagreement to flag")
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=30,
        help="Pairs recovered from fewer inliers are unusable evidence, not disagreements",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    poses = np.load(args.poses)["data"].astype(np.float64)
    intrinsics = np.load(args.intrinsics)["data"].astype(np.float64)
    focal, cx, cy = float(intrinsics[0, 0]), float(intrinsics[0, 2]), float(intrinsics[0, 3])

    frames = sorted(args.images.glob("*.jpg"))
    if len(frames) != len(poses):
        print(f"{len(frames)} images but {len(poses)} poses", file=sys.stderr)
        return 1

    solved = [
        rotation_angle(poses[index + 1, :3, :3] @ poses[index, :3, :3].T) for index in range(len(poses) - 1)
    ]

    tasks = [
        (str(frames[index]), str(frames[index + 1]), focal, cx, cy, index, args.scale)
        for index in range(len(frames) - 1)
    ]
    measured: dict[int, float] = {}
    inliers: dict[int, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, angle, inlier_count in pool.map(image_rotation, tasks):
            measured[index] = angle
            inliers[index] = inlier_count

    rows = []
    for index, solved_angle in enumerate(solved):
        image_angle = measured[index]
        rows.append(
            {
                "pair": [index, index + 1],
                "solved_deg": solved_angle,
                "image_deg": image_angle,
                "error_deg": abs(solved_angle - image_angle),
                "inliers": inliers[index],
            }
        )

    valid = [
        row for row in rows if np.isfinite(row["error_deg"]) and row["inliers"] >= args.min_inliers
    ]
    errors = np.array([row["error_deg"] for row in valid])
    flagged = sorted(
        (row for row in valid if row["error_deg"] > args.tolerance),
        key=lambda row: row["error_deg"],
        reverse=True,
    )

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "poses": str(args.poses),
        "pairs_checked": len(valid),
        "pairs_unusable": len(rows) - len(valid),
        "error_deg": {
            "median": float(np.median(errors)) if len(errors) else None,
            "mean": float(errors.mean()) if len(errors) else None,
            "p95": float(np.percentile(errors, 95)) if len(errors) else None,
            "max": float(errors.max()) if len(errors) else None,
        },
        "tolerance_deg": args.tolerance,
        "disagreeing_pairs": len(flagged),
        "worst": flagged[:10],
    }

    print(json.dumps(document, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
