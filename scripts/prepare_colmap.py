#!/usr/bin/env python3
"""Normalize VIPE's COLMAP text export for Nerfstudio's COLMAP parser."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")


def data_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def rewrite_images_txt(source: Path, destination: Path) -> list[str]:
    referenced: list[str] = []
    output: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            output.append(raw)
            continue

        fields = stripped.split()
        # COLMAP image records contain 9 scalar fields followed by IMAGE_NAME.
        # POINTS2D records contain triples and therefore cannot have 10 fields.
        if len(fields) == 10:
            try:
                int(fields[0])
                int(fields[8])
            except ValueError:
                output.append(raw)
                continue
            image_name = Path(fields[9]).name
            fields[9] = image_name
            referenced.append(image_name)
            output.append(" ".join(fields))
        else:
            output.append(raw)

    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return referenced


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="One sequence produced by vipe_to_colmap.py")
    parser.add_argument("--output", required=True, type=Path, help="New Nerfstudio dataset directory")
    args = parser.parse_args()

    source_images = args.source / "images"
    if not source_images.is_dir():
        parser.error(f"source images directory is missing: {source_images}")
    for filename in MODEL_FILES:
        if not (args.source / filename).is_file():
            parser.error(f"source model file is missing: {args.source / filename}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-colmap-", dir=args.output.parent) as temp_name:
        temporary = Path(temp_name)
        output_images = temporary / "images"
        sparse = temporary / "sparse" / "0"
        output_images.mkdir(parents=True)
        sparse.mkdir(parents=True)

        referenced = rewrite_images_txt(args.source / "images.txt", sparse / "images.txt")
        if not referenced:
            raise RuntimeError("no COLMAP image records found")
        if len(referenced) != len(set(referenced)):
            raise RuntimeError("duplicate image names in COLMAP images.txt")

        source_by_name = {path.name: path for path in source_images.iterdir() if path.is_file()}
        missing = sorted(set(referenced) - source_by_name.keys())
        if missing:
            raise RuntimeError(f"{len(missing)} referenced images are missing; first: {missing[0]}")
        for name in referenced:
            link_or_copy(source_by_name[name], output_images / name)

        shutil.copy2(args.source / "cameras.txt", sparse / "cameras.txt")
        shutil.copy2(args.source / "points3D.txt", sparse / "points3D.txt")

        cameras = data_lines(sparse / "cameras.txt")
        points = data_lines(sparse / "points3D.txt")
        if not cameras:
            raise RuntimeError("cameras.txt contains no cameras")
        if not points:
            raise RuntimeError("points3D.txt contains no initialization points")
        if len(list(output_images.iterdir())) != len(referenced):
            raise RuntimeError("normalized image count does not match COLMAP image count")

        temporary.rename(args.output)

    print(f"COLMAP dataset ready: {args.output}")
    print(f"cameras={len(cameras)} images={len(referenced)} points={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
