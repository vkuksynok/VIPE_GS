#!/usr/bin/env python3
"""Build a Nerfstudio camera path from the trajectory the model was trained on.

The path is taken from the trained run's own dataparser output, so it already
lives in the model's world space with the model's conventions; nothing is
converted by hand. Poses are interpolated so the flight renders smoothly
instead of stepping once per captured second.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from nerfstudio.cameras.camera_paths import get_interpolated_camera_path


def load_cameras(config_path: Path):
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    dataparser_config = config.pipeline.datamanager.dataparser
    if hasattr(dataparser_config, "eval_mode"):
        # "all" puts every frame in the train split, so the path follows the
        # whole flight rather than only the frames used for optimization.
        dataparser_config.eval_mode = "all"
    outputs = dataparser_config.setup().get_dataparser_outputs("train")
    return outputs.cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Trained run's config.yml")
    parser.add_argument("--output", required=True, type=Path, help="Camera path JSON to write")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5, help="Interpolated cameras per captured pair")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1440)
    args = parser.parse_args()

    cameras = load_cameras(args.config)
    if len(cameras) < 2:
        raise RuntimeError(f"need at least two cameras, found {len(cameras)}")

    path = get_interpolated_camera_path(cameras, steps=args.steps, order_poses=False)

    # get_path_from_json rebuilds the focal length from a vertical field of view.
    focal_y = float(cameras.fy[0].item())
    fov = math.degrees(2.0 * math.atan(args.height / (2.0 * focal_y)))

    entries = []
    for index in range(len(path)):
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[:3, :4] = path.camera_to_worlds[index].to(torch.float64)
        entries.append(
            {
                "camera_to_world": [float(value) for value in matrix.flatten().tolist()],
                "fov": fov,
                "aspect": args.width / args.height,
            }
        )

    document = {
        "camera_type": "perspective",
        "render_height": args.height,
        "render_width": args.width,
        "fps": args.fps,
        "seconds": len(entries) / args.fps,
        "is_cycle": False,
        "smoothness_value": 0.0,
        "camera_path": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(
        f"captured cameras: {len(cameras)}\n"
        f"path cameras: {len(entries)}\n"
        f"fov: {fov:.3f} deg\n"
        f"duration: {document['seconds']:.1f} s at {args.fps} fps\n"
        f"written: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
