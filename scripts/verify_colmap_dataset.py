#!/usr/bin/env python3
"""Verify a COLMAP text dataset before it is handed to Nerfstudio.

Checks the structure Nerfstudio's COLMAP dataparser expects, then confirms the
reconstruction is geometrically coherent: cameras must see the point cloud, and
the recovered camera track must stay continuous.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


CONTINUITY_SIGMAS = 10.0


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_records(path: Path) -> list[list[str]]:
    return [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--expected-images", type=int, default=126)
    parser.add_argument("--sample-points", type=int, default=60000)
    parser.add_argument("--min-visible-fraction", type=float, default=0.05)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    problems: list[str] = []
    model = args.dataset / "sparse" / "0"
    images_dir = args.dataset / "images"
    for required in (model / "cameras.txt", model / "images.txt", model / "points3D.txt", images_dir):
        if not required.exists():
            problems.append(f"missing {required}")
    if problems:
        print("\n".join(f"PROBLEM: {problem}" for problem in problems), file=sys.stderr)
        return 1

    cameras = read_records(model / "cameras.txt")
    if len(cameras) != 1:
        problems.append(f"expected a single shared camera, found {len(cameras)}")
    camera_id, camera_model, width, height = cameras[0][0], cameras[0][1], int(cameras[0][2]), int(cameras[0][3])
    if camera_model != "PINHOLE":
        problems.append(f"camera model is {camera_model}, Splatfacto expects PINHOLE")
    focal_x, focal_y, center_x, center_y = (float(value) for value in cameras[0][4:8])
    intrinsics = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]])

    records = [fields for fields in read_records(model / "images.txt") if len(fields) == 10]
    if len(records) != args.expected_images:
        problems.append(f"images.txt holds {len(records)} poses, expected {args.expected_images}")

    names = [fields[9] for fields in records]
    if len(set(names)) != len(names):
        problems.append("images.txt references a file more than once")
    if any(Path(name).parent != Path("") for name in names):
        problems.append("image names carry a directory prefix, which Nerfstudio would duplicate")

    on_disk = {path.name for path in images_dir.glob("*.jpg")}
    missing = [name for name in names if name not in on_disk]
    if missing:
        problems.append(f"{len(missing)} referenced images are absent, first: {missing[0]}")
    unreferenced = sorted(on_disk - set(names))
    if unreferenced:
        problems.append(f"{len(unreferenced)} images are not referenced, first: {unreferenced[0]}")

    with Image.open(images_dir / names[0]) as sample:
        if sample.size != (width, height):
            problems.append(f"camera says {width}x{height}, image is {sample.width}x{sample.height}")

    quaternions = np.array([[float(value) for value in fields[1:5]] for fields in records])
    translations = np.array([[float(value) for value in fields[5:8]] for fields in records])
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-3):
        problems.append("images.txt contains a non-unit quaternion")
    if len({fields[8] for fields in records}) != 1 or records[0][8] != camera_id:
        problems.append("not every image points at the shared camera id")

    rotations = np.array([quaternion_to_rotation(quaternion) for quaternion in quaternions])
    # COLMAP stores world-to-camera, so the centre is -R^T t.
    centers = np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), -translations)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median_step = float(np.median(steps))
    scaled_mad = 1.4826 * float(np.median(np.abs(steps - median_step)))
    threshold = median_step + CONTINUITY_SIGMAS * scaled_mad if scaled_mad > 0 else float("inf")
    jumps = [
        {"pair": [index, index + 1], "step": float(step)}
        for index, step in enumerate(steps)
        if step > threshold
    ]
    if jumps:
        problems.append(f"{len(jumps)} discontinuous camera step(s) above {threshold:.3f}")

    point_records = read_records(model / "points3D.txt")
    points = np.array([[float(value) for value in fields[1:4]] for fields in point_records])
    if not np.isfinite(points).all():
        problems.append("points3D.txt contains non-finite coordinates")

    generator = np.random.default_rng(0)
    if len(points) > args.sample_points:
        sample = points[generator.choice(len(points), args.sample_points, replace=False)]
    else:
        sample = points

    visible: list[float] = []
    for rotation, translation in zip(rotations, translations):
        camera_points = sample @ rotation.T + translation
        in_front = camera_points[:, 2] > 0
        projected = (camera_points[in_front] @ intrinsics.T)[:, :2] / camera_points[in_front][:, 2:3]
        inside = (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        visible.append(float(inside.sum()) / len(sample))

    visible_array = np.array(visible)
    if float(visible_array.min()) < args.min_visible_fraction:
        problems.append(
            f"a camera sees only {visible_array.min():.1%} of the point cloud,"
            f" below the {args.min_visible_fraction:.0%} floor"
        )

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "camera": {
            "model": camera_model,
            "width": width,
            "height": height,
            "fx": focal_x,
            "fy": focal_y,
            "cx": center_x,
            "cy": center_y,
        },
        "images": len(records),
        "points": len(points),
        "track": {
            "step_median": median_step,
            "step_max": float(steps.max()),
            "continuity_threshold": threshold,
            "discontinuities": jumps,
            "path_length": float(steps.sum()),
        },
        "visibility": {
            "min": float(visible_array.min()),
            "median": float(np.median(visible_array)),
            "max": float(visible_array.max()),
        },
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
