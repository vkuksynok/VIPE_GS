#!/usr/bin/env python3
"""Check a prepared sequence: ordering, count, aspect ratio, frame traceability."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ffprobe_stream(video: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height,avg_frame_rate,sample_aspect_ratio",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=126)
    parser.add_argument(
        "--dataset-report",
        type=Path,
        help="verify_dataset.py JSON report; enables hash traceability to the source archive",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.prepared_dir.is_dir():
        parser.error(f"prepared directory not found: {args.prepared_dir}")

    problems: list[str] = []
    manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = read_rows(args.prepared_dir / "frames.csv")
    video = args.prepared_dir / manifest["video"]
    images = sorted((args.prepared_dir / "images").glob("*.jpg"))

    if len(rows) != args.expected_count:
        problems.append(f"frames.csv holds {len(rows)} rows, expected {args.expected_count}")
    if len(images) != args.expected_count:
        problems.append(f"images/ holds {len(images)} files, expected {args.expected_count}")

    indices = [int(row["frame_index"]) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        problems.append("frame_index values are not a contiguous 1..N sequence")

    expected_names = [f"frame_{index:06d}.jpg" for index in indices]
    if [row["prepared_name"] for row in rows] != expected_names:
        problems.append("prepared_name values do not follow frame_%06d.jpg in order")
    if [image.name for image in images] != sorted(expected_names):
        problems.append("images/ contents do not match the names in frames.csv")

    timestamps = [row["timestamp_utc"] for row in rows]
    if timestamps != sorted(timestamps):
        problems.append("rows are not ordered by capture timestamp")

    source_names = [row["source_name"] for row in rows]
    if len(set(source_names)) != len(source_names):
        problems.append("a source photo is used by more than one frame index")

    stream = ffprobe_stream(video)
    encoded = int(stream["nb_read_frames"])
    width, height = int(stream["width"]), int(stream["height"])
    if encoded != args.expected_count:
        problems.append(f"video holds {encoded} frames, expected {args.expected_count}")

    traceability: dict[str, object] = {"checked": False}
    if args.dataset_report is not None:
        report = json.loads(args.dataset_report.read_text(encoding="utf-8"))
        source_hashes: dict[str, str] = {}
        source_aspects: set[float] = set()
        for archive in report["archives"].values():
            for frame in archive["frames"]:
                source_hashes.setdefault(frame["name"], frame["sha256"])
                source_aspects.add(round(frame["width"] / frame["height"], 6))
        if source_aspects and round(width / height, 6) not in source_aspects:
            problems.append(
                f"video aspect {round(width / height, 6)} matches no source aspect {sorted(source_aspects)}"
            )
        unmatched = [row["source_name"] for row in rows if row["source_name"] not in source_hashes]
        mismatched = [
            row["source_name"]
            for row in rows
            if row["source_name"] in source_hashes and row["sha256"] != source_hashes[row["source_name"]]
        ]
        if unmatched:
            problems.append(f"{len(unmatched)} frames are not present in the dataset report")
        if mismatched:
            problems.append(f"{len(mismatched)} frames differ in SHA-256 from the source archive")
        traceability = {
            "checked": True,
            "matched": len(rows) - len(unmatched) - len(mismatched),
            "unmatched": unmatched[:5],
            "mismatched": mismatched[:5],
        }

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prepared_dir": str(args.prepared_dir),
        "manifest": manifest,
        "video": {
            "name": video.name,
            "frames": encoded,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 6),
            "avg_frame_rate": stream["avg_frame_rate"],
            "bytes": video.stat().st_size,
        },
        "first_source_name": source_names[0] if source_names else None,
        "last_source_name": source_names[-1] if source_names else None,
        "traceability": traceability,
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
