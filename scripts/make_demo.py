#!/usr/bin/env python3
"""Assemble the end-to-end demo video from the pipeline's own outputs.

Title cards carry the measured numbers rather than claims, and the footage is
the real captured frames, the held-out comparison, and the rendered flight.
Segments are encoded separately with identical parameters and concatenated, so
sources of different sizes and frame rates can be mixed safely.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 1440, 1080, 30
BACKGROUND = (16, 18, 22)
FOREGROUND = (238, 240, 244)
ACCENT = (122, 176, 255)
MUTED = (150, 158, 170)

FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def load_font(size: int, explicit: str | None) -> ImageFont.FreeTypeFont:
    candidates = (explicit,) + FONT_CANDIDATES if explicit else FONT_CANDIDATES
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_card(path: Path, title: str, lines: list[str], font_path: str | None, footer: str = "") -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(64, font_path)
    body_font = load_font(38, font_path)
    footer_font = load_font(28, font_path)

    draw.text((110, 250), title, font=title_font, fill=FOREGROUND)
    draw.line((110, 350, WIDTH - 110, 350), fill=ACCENT, width=3)

    y = 420
    for line in lines:
        colour = MUTED if line.startswith("  ") else FOREGROUND
        draw.text((110, y), line.strip() if colour is FOREGROUND else line, font=body_font, fill=colour)
        y += 62

    if footer:
        draw.text((110, HEIGHT - 130), footer, font=footer_font, fill=MUTED)
    image.save(path)


def draw_comparison(path: Path, source: Path, font_path: str | None) -> None:
    """Place a side-by-side reference/render pair on a captioned card."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, font_path)
    label_font = load_font(32, font_path)
    note_font = load_font(28, font_path)

    with Image.open(source) as pair:
        scaled = pair.convert("RGB")
        ratio = (WIDTH - 160) / scaled.width
        scaled = scaled.resize((int(scaled.width * ratio), int(scaled.height * ratio)), Image.LANCZOS)
    top = (HEIGHT - scaled.height) // 2
    image.paste(scaled, (80, top))

    draw.text((110, 120), "Held-out frame: reference vs render", font=title_font, fill=FOREGROUND)
    draw.text((100, top - 46), "reference", font=label_font, fill=ACCENT)
    draw.text((WIDTH // 2 + 40, top - 46), "render", font=label_font, fill=ACCENT)
    draw.text(
        (110, top + scaled.height + 40),
        "Never optimized: every 8th frame was held out. PSNR 15.13 here, 22.29 on training views.",
        font=note_font,
        fill=MUTED,
    )
    image.save(path)


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise RuntimeError(f"command failed: {' '.join(command[:3])} ...")


def encode_still(ffmpeg: str, still: Path, seconds: float, destination: Path, crf: int) -> None:
    run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-t", f"{seconds}", "-i", str(still),
            "-vf", f"scale={WIDTH}:{HEIGHT},format=yuv420p",
            "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            str(destination),
        ]
    )


def encode_sequence(ffmpeg: str, pattern: str, rate: float, destination: Path, crf: int) -> None:
    run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", f"{rate}", "-i", pattern,
            "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p",
            "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            str(destination),
        ]
    )


def encode_clip(ffmpeg: str, source: Path, destination: Path, crf: int) -> None:
    run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"scale={WIDTH}:{HEIGHT},format=yuv420p",
            "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-an", str(destination),
        ]
    )


def extract_frames(archive_or_dir: Path, destination: Path, limit: int | None) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_or_dir.is_dir():
        names = sorted(p for p in archive_or_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
        if limit:
            names = names[:limit]
        for index, source in enumerate(names, 1):
            shutil.copyfile(source, destination / f"{index:06d}.jpg")
        return len(names)

    with zipfile.ZipFile(archive_or_dir) as archive:
        members = sorted(
            (info for info in archive.infolist() if not info.is_dir()),
            key=lambda info: PurePosixPath(info.filename).name.lower(),
        )
        members = [m for m in members if PurePosixPath(m.filename).suffix.lower() in {".jpg", ".jpeg"}]
        if limit:
            members = members[:limit]
        for index, member in enumerate(members, 1):
            with archive.open(member) as handle, (destination / f"{index:06d}.jpg").open("wb") as out:
                shutil.copyfileobj(handle, out)
        return len(members)


def probe_frames(ffprobe: str, video: Path) -> int:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of",
            "default=nokey=1:noprint_wrappers=1", str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path, help="Source photo ZIP or directory")
    parser.add_argument("--flythrough", required=True, type=Path, help="Rendered camera-path MP4")
    parser.add_argument("--comparison", type=Path, help="Side-by-side reference/render PNG")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--font", help="TTF to draw the cards with")
    parser.add_argument("--slideshow-fps", type=float, default=12.0)
    parser.add_argument("--crf", type=int, default=24, help="x264 quality; higher is smaller")
    parser.add_argument("--repo", default="", help="Repository URL for the title card")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    for required in (args.frames, args.flythrough):
        if not required.exists():
            parser.error(f"missing input: {required}")

    with tempfile.TemporaryDirectory(prefix="gs-demo-") as workspace:
        work = Path(workspace)
        segments: list[Path] = []

        def add_card(name: str, seconds: float, title: str, lines: list[str], footer: str = "") -> None:
            still = work / f"{name}.png"
            draw_card(still, title, lines, args.font, footer)
            clip = work / f"{name}.mp4"
            encode_still(args.ffmpeg, still, seconds, clip, args.crf)
            segments.append(clip)

        add_card(
            "00_title", 4.5,
            "VIPE to Gaussian Splatting",
            [
                "126 drone photos of an abandoned industrial site",
                "NVIDIA VIPE  ->  COLMAP  ->  Nerfstudio Splatfacto",
                "RTX PRO 4500 Blackwell, about 1 h 15 min of GPU time",
            ],
            args.repo,
        )

        frame_dir = work / "frames"
        count = extract_frames(args.frames, frame_dir, limit=None)
        slideshow = work / "01_input.mp4"
        encode_sequence(args.ffmpeg, str(frame_dir / "%06d.jpg"), args.slideshow_fps, slideshow, args.crf)
        segments.append(slideshow)

        add_card(
            "02_vipe", 6.0,
            "Camera solve with VIPE",
            [
                f"{count} frames at 1920x1440, captured at 1 fps",
                "126 poses in 353 s, peak 24,806 MiB",
                "Median rotation error vs the images: 0.26 deg",
            ],
            "Verified against SIFT/essential-matrix estimates, not trusted blindly",
        )

        add_card(
            "03_defect", 7.5,
            "One pose was wrong",
            [
                "Frames 22-23: VIPE solved 64.74 deg of rotation",
                "  The images show 4.64 deg. The camera never turned sharply.",
                "Four VIPE configurations moved the defect, none removed it",
                "Re-solved with RGB-D PnP from VIPE's own metric depth",
                "  Disagreeing pairs 1 -> 0, worst error 60.79 -> 3.37 deg",
            ],
            "Median and 95th percentile unchanged: the repair touched only the defect",
        )

        add_card(
            "04_training", 6.0,
            "Splatfacto training",
            [
                "30,000 iterations in 43 min 50 s, peak 16,694 MiB",
                "7,239,301 gaussians; 7,055,547 exported to a 1.75 GB PLY",
                "Every 8th frame held out of optimization",
            ],
            "Blackwell needed torch 2.9.0+cu128; the documented pins stop at sm_90",
        )

        if args.comparison and args.comparison.is_file():
            still = work / "05_comparison.png"
            draw_comparison(still, args.comparison, args.font)
            clip = work / "05_comparison.mp4"
            encode_still(args.ffmpeg, still, 7.0, clip, args.crf)
            segments.append(clip)

        add_card(
            "06_flight", 4.0,
            "Rendered camera path",
            [
                "625 cameras interpolated from the 126 captured poses",
                "1920x1440, 30 fps, 20.8 s",
                "Most frames are viewpoints that were never captured",
            ],
        )

        flight = work / "07_flight.mp4"
        encode_clip(args.ffmpeg, args.flythrough, flight, args.crf)
        segments.append(flight)

        add_card(
            "08_end", 7.0,
            "Honest limits",
            [
                "Held-out PSNR 15.13 vs 22.29 on training views",
                "  The model reproduces what it saw; new viewpoints are weaker",
                "Causes: vegetation detail, sparse views, 1 fps baselines, pose error",
                "Untried lever: Splatfacto's camera optimizer (SO3xR3)",
            ],
            "Every defect and how it was found: docs/TROUBLESHOOTING.md",
        )

        listing = work / "segments.txt"
        listing.write_text(
            "\n".join(f"file '{segment.as_posix()}'" for segment in segments) + "\n",
            encoding="utf-8",
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(args.output),
            ]
        )

    frames = probe_frames(args.ffprobe, args.output)
    document = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "frames": frames,
        "seconds": round(frames / FPS, 2),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "crf": args.crf,
        "source_photos": count,
        "segments": len(segments),
    }
    print(json.dumps(document, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
