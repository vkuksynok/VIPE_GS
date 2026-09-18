#!/usr/bin/env python3
"""Create an ordered frame directory, manifest, and MP4 from DJI JPEGs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


DJI_NAME = re.compile(
    r"^dji_(?P<timestamp>\d{14})_(?P<sequence>\d+)_v\.jpe?g$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceFrame:
    name: str
    timestamp: datetime
    sequence: int
    opener: object

    @property
    def sort_key(self) -> tuple[datetime, int, str]:
        return self.timestamp, self.sequence, self.name.lower()


def parse_name(name: str) -> tuple[datetime, int] | None:
    match = DJI_NAME.fullmatch(name)
    if not match:
        return None
    timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return timestamp, int(match.group("sequence"))


def directory_frames(path: Path) -> list[SourceFrame]:
    frames: list[SourceFrame] = []
    for image in path.rglob("*"):
        if not image.is_file():
            continue
        parsed = parse_name(image.name)
        if parsed is None:
            continue
        timestamp, sequence = parsed
        frames.append(SourceFrame(image.name, timestamp, sequence, image))
    return sorted(frames, key=lambda frame: frame.sort_key)


def zip_frames(archive: zipfile.ZipFile) -> list[SourceFrame]:
    frames: list[SourceFrame] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = PurePosixPath(info.filename).name
        parsed = parse_name(name)
        if parsed is None:
            continue
        timestamp, sequence = parsed
        frames.append(SourceFrame(name, timestamp, sequence, info))
    return sorted(frames, key=lambda frame: frame.sort_key)


def copy_with_hash(source: BinaryIO, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ffprobe_frame_count(video: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def unique_names(frames: Iterable[SourceFrame]) -> bool:
    names = [frame.name.lower() for frame in frames]
    return len(names) == len(set(names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ZIP archive or image directory")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--video-name", default="zavod70.mp4")
    parser.add_argument(
        "--limit",
        type=int,
        help="Keep only the first N frames after ordering; used for smoke tests",
    )
    args = parser.parse_args()

    if args.fps <= 0 or args.max_width <= 0:
        parser.error("--fps and --max-width must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        parser.error("ffmpeg and ffprobe must be installed")

    archive: zipfile.ZipFile | None = None
    if args.input.is_file() and args.input.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(args.input)
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad}")
        frames = zip_frames(archive)
    elif args.input.is_dir():
        frames = directory_frames(args.input)
    else:
        parser.error("input must be a ZIP archive or directory")

    if not frames:
        raise RuntimeError("no DJI JPEG frames found")
    if not unique_names(frames):
        raise RuntimeError("duplicate source basenames found")

    if args.limit is not None:
        if args.limit > len(frames):
            raise RuntimeError(f"--limit {args.limit} exceeds the {len(frames)} available frames")
        frames = frames[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-zavod70-", dir=args.output.parent) as temp_name:
        temporary = Path(temp_name)
        images = temporary / "images"
        images.mkdir()
        rows: list[dict[str, object]] = []

        for index, frame in enumerate(frames, 1):
            destination = images / f"frame_{index:06d}.jpg"
            if archive is not None:
                with archive.open(frame.opener) as source:
                    sha256, size = copy_with_hash(source, destination)
            else:
                with Path(frame.opener).open("rb") as source:
                    sha256, size = copy_with_hash(source, destination)
            rows.append(
                {
                    "frame_index": index,
                    "prepared_name": destination.name,
                    "source_name": frame.name,
                    "timestamp_utc": frame.timestamp.isoformat(),
                    "sequence": frame.sequence,
                    "bytes": size,
                    "sha256": sha256,
                }
            )

        if archive is not None:
            archive.close()

        csv_path = temporary / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        video = temporary / args.video_name
        scale = f"scale='min({args.max_width},iw)':-2"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(images / "frame_%06d.jpg"),
                "-vf",
                scale,
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ]
        )
        encoded_frames = ffprobe_frame_count(video)
        if encoded_frames != len(rows):
            raise RuntimeError(f"encoded {encoded_frames} frames, expected {len(rows)}")

        manifest = {
            "source": str(args.input.resolve()),
            "frame_count": len(rows),
            "encoded_frame_count": encoded_frames,
            "limit": args.limit,
            "fps": args.fps,
            "max_width": args.max_width,
            "first_source_name": rows[0]["source_name"],
            "last_source_name": rows[-1]["source_name"],
            "video": video.name,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.rename(args.output)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
