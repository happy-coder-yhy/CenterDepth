"""Preview video compression helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


PREVIEW_VIDEO_BITRATE = 3_000_000


def preview_video_name(video_name: str) -> str:
    """Return the sibling filename used for the compressed preview."""
    path = Path(video_name)
    return f"{path.stem}_preview{path.suffix}"


def build_preview_compression_command(
    input_path: str | Path,
    output_path: str | Path,
    bitrate: int = PREVIEW_VIDEO_BITRATE,
    *,
    fps: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    """Build an ffmpeg command for a fixed-bitrate preview transcode."""
    if bitrate < 1:
        raise ValueError(f"Preview video bitrate must be positive, got {bitrate}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-b:v", f"{bitrate // 1000}k",
        "-minrate", f"{bitrate // 1000}k",
        "-maxrate", f"{bitrate // 1000}k",
        "-bufsize", f"{bitrate * 2 // 1000}k",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
    ]
    if fps:
        command += ["-r", fps]
    if width is not None and height is not None:
        command += ["-s", f"{width}x{height}"]
    command.append(str(output_path))
    return command


def probe_preview_video(path: str | Path) -> dict[str, int | str]:
    """Read the first video stream's geometry and frame-rate expression."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("压缩深度视频需要 ffprobe，但当前环境未找到 ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"深度视频没有可用视频流: {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": str(stream["r_frame_rate"]),
    }


def compress_preview_video(
    path: str | Path,
    bitrate: int = PREVIEW_VIDEO_BITRATE,
    *,
    output_path: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Compress an MP4, optionally writing the preview to a second file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"待压缩深度视频不存在: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("压缩深度视频需要 ffmpeg，但当前环境未找到 ffmpeg")
    target_path = Path(output_path) if output_path is not None else input_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = probe_preview_video(input_path)
    temporary_path = target_path.with_name(
        f".{target_path.stem}.compressed{target_path.suffix}"
    )
    command = build_preview_compression_command(
        input_path,
        temporary_path,
        bitrate,
        fps=str(metadata["fps"]),
        width=int(metadata["width"]),
        height=int(metadata["height"]),
    )
    try:
        subprocess.run(command, check=True)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg 未生成有效压缩视频: {temporary_path}")
        temporary_path.replace(target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "target_bitrate_bps": int(bitrate),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "fps": str(metadata["fps"]),
    }
