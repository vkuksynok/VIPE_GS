#!/usr/bin/env python3
"""Verify dataset archives on persistent storage and materialise raw frames."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image


DJI_NAME = re.compile(
    r"^dji_(?P<timestamp>\d{14})_(?P<sequence>\d+)_v\.jpe?g$",
    re.IGNORECASE,
)
CHUNK = 1024 * 1024
# DJI writes multi-picture (MPO) containers for the full-resolution stills. They are
# JPEG-compatible and Pillow, ffmpeg, and COLMAP all read the primary image.
ACCEPTED_FORMATS = frozenset({"JPEG", "MPO"})


@dataclass(frozen=True)
class FrameReport:
    name: str
    sequence: int
    timestamp_utc: str
    bytes: int
    sha256: str
    width: int
    height: int
    format: str


def parse_name(name: str) -> tuple[datetime, int] | None:
    match = DJI_NAME.fullmatch(name)
    if not match:
        return None
    timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return timestamp, int(match.group("sequence"))


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def decode_frame(payload: bytes) -> tuple[int, int, str]:
    """Structurally verify and fully decode a JPEG, returning its geometry."""
    Image.open(io.BytesIO(payload)).verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.width, image.height, image.format or "UNKNOWN"


def inspect_archive(path: Path, expected_count: int) -> dict[str, object]:
    problems: list[str] = []
    archive_sha256 = file_digest(path)[0]

    frames: list[FrameReport] = []
    ignored: list[str] = []
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            problems.append(f"corrupt ZIP member: {corrupt}")

        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            parsed = parse_name(name)
            if parsed is None:
                ignored.append(info.filename)
                continue
            timestamp, sequence = parsed
            payload = archive.read(info)
            try:
                width, height, image_format = decode_frame(payload)
            except Exception as error:  # noqa: BLE001 - reported, not raised
                problems.append(f"unreadable image {name}: {error}")
                continue
            frames.append(
                FrameReport(
                    name=name,
                    sequence=sequence,
                    timestamp_utc=timestamp.isoformat(),
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    width=width,
                    height=height,
                    format=image_format,
                )
            )

    frames.sort(key=lambda frame: (frame.timestamp_utc, frame.sequence, frame.name.lower()))

    lowered = [frame.name.lower() for frame in frames]
    duplicates = sorted({name for name in lowered if lowered.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate basenames: {', '.join(duplicates)}")

    sequences = [frame.sequence for frame in frames]
    missing_sequences: list[int] = []
    if sequences:
        missing_sequences = sorted(set(range(min(sequences), max(sequences) + 1)) - set(sequences))
    if missing_sequences:
        problems.append(f"missing sequence numbers: {missing_sequences}")

    if expected_count and len(frames) != expected_count:
        problems.append(f"expected {expected_count} frames, found {len(frames)}")

    formats = sorted({frame.format for frame in frames})
    unsupported = sorted(set(formats) - ACCEPTED_FORMATS)
    if unsupported:
        problems.append(f"unexpected image formats: {', '.join(unsupported)}")

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": archive_sha256,
        "frame_count": len(frames),
        "resolutions": sorted({f"{frame.width}x{frame.height}" for frame in frames}),
        "formats": formats,
        "sequence_min": min(sequences) if sequences else None,
        "sequence_max": max(sequences) if sequences else None,
        "missing_sequences": missing_sequences,
        "duplicate_names": duplicates,
        "ignored_members": ignored,
        "first_frame": frames[0].name if frames else None,
        "last_frame": frames[-1].name if frames else None,
        "frames": [asdict(frame) for frame in frames],
        "problems": problems,
    }


def extract_archive(path: Path, destination: Path, report: dict[str, object]) -> dict[str, object]:
    """Extract image members flat into destination, leaving the archive untouched."""
    expected = {frame["name"]: frame["sha256"] for frame in report["frames"]}
    destination.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    written = 0

    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if name not in expected:
                continue
            target = destination / name
            digest = hashlib.sha256()
            with archive.open(info) as source, target.open("wb") as handle:
                while chunk := source.read(CHUNK):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != expected[name]:
                problems.append(f"checksum mismatch after extraction: {name}")
            written += 1

    if written != len(expected):
        problems.append(f"extracted {written} of {len(expected)} frames")

    return {"destination": str(destination), "extracted": written, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=126)
    parser.add_argument("--extract", type=Path, help="Archive to extract; must also be passed with --archive")
    parser.add_argument("--extract-to", type=Path, help="Directory receiving the extracted frames")
    parser.add_argument("--report", type=Path, help="Write the full JSON report here")
    args = parser.parse_args()

    if (args.extract is None) != (args.extract_to is None):
        parser.error("--extract and --extract-to must be used together")
    for archive in args.archive:
        if not archive.is_file():
            parser.error(f"archive not found: {archive}")
    if args.extract is not None and args.extract not in args.archive:
        parser.error("--extract must name one of the --archive paths")

    reports: dict[str, dict[str, object]] = {}
    for archive in args.archive:
        print(f"== {archive}", flush=True)
        report = inspect_archive(archive, args.expected_count)
        reports[str(archive)] = report
        print(
            f"   frames={report['frame_count']}"
            f" resolutions={','.join(report['resolutions'])}"
            f" formats={','.join(report['formats'])}"
            f" sha256={report['sha256']}",
            flush=True,
        )
        for problem in report["problems"]:
            print(f"   PROBLEM: {problem}", flush=True)

    extraction: dict[str, object] = {}
    if args.extract is not None and args.extract_to is not None:
        print(f"== extracting {args.extract} -> {args.extract_to}", flush=True)
        extraction = extract_archive(args.extract, args.extract_to, reports[str(args.extract)])
        print(f"   extracted={extraction['extracted']}", flush=True)
        for problem in extraction["problems"]:
            print(f"   PROBLEM: {problem}", flush=True)

    document = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_count": args.expected_count,
        "archives": reports,
        "extraction": extraction,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(f"== report written: {args.report}", flush=True)

    problems = [problem for report in reports.values() for problem in report["problems"]]
    problems.extend(extraction.get("problems", []))
    if problems:
        print(f"FAILED with {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
