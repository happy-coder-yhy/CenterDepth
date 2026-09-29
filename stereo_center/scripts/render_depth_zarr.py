"""Read-only adaptive log-depth visualization with a fixed 0.3 m lower bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center.depth_video_visualization import render_adaptive_frames  # noqa: E402
from stereo_center.video_compression import compress_preview_video, preview_video_name  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-zarr", type=Path, required=True)
    parser.add_argument("--depth-key", default="depth")
    parser.add_argument("--valid-key", help="Optional saved validity mask array")
    parser.add_argument("--fps", type=float, default=None, help="Required if Zarr has no fps metadata")
    parser.add_argument("--dmax-m", type=float, default=20.0, help="Adaptive display ceiling in meters (default 20)")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.dmax_m) or args.dmax_m < 1.0:
        parser.error("--dmax-m must be finite and >= 1 meter")
    # Register the current codec for reading only; no writer or compression change.
    try:
        import stereo_center.depth_sz3  # noqa: F401
    except ImportError:
        pass  # Lossless Zarr inputs do not require the optional SZ3 dependency.
    group = zarr.open_group(str(args.depth_zarr), mode="r")
    depth = group[args.depth_key]
    valid = group[args.valid_key] if args.valid_key else None
    if len(depth.shape) != 3 or not depth.shape[0] or min(depth.shape[1:]) <= 0:
        raise ValueError("Expected nonempty (frames, height, width) depth")
    if valid is not None and valid.shape != depth.shape:
        raise ValueError("Validity array shape must match depth")
    fps = args.fps if args.fps is not None else group.attrs.get("fps", depth.attrs.get("fps"))
    if fps is None or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("A positive finite --fps or fps metadata is required")
    args.outdir.mkdir(parents=True, exist_ok=True)
    video = args.outdir / "depth_adaptive.mp4"
    preview = args.outdir / preview_video_name(video.name)
    metadata_path = args.outdir / "depth_color_scale.json"
    if any(p.exists() for p in (video, preview, metadata_path)):
        raise FileExistsError("Use a new output directory; refusing to replace existing results")

    def frames():
        # Decode each physical time chunk once, including SZ3 32-frame chunks.
        for start in range(0, depth.shape[0], depth.chunks[0]):
            block = depth[start:start + depth.chunks[0]]
            masks = valid[start:start + depth.chunks[0]] if valid is not None else None
            for i, frame in enumerate(block):
                yield frame, masks[i] if masks is not None else np.isfinite(frame) & (frame > 0)

    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                             (depth.shape[2], depth.shape[1]))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {video}")
    samples = {min(n, depth.shape[0] - 1) for n in [100, 400, 700]}

    def write_frame(index, image):
        writer.write(image)
        if index in samples:
            if not cv2.imwrite(str(args.outdir / f"frame_{index:05d}.png"), image):
                raise RuntimeError("Failed to save representative frame")

    try:
        metadata = render_adaptive_frames(frames, write_frame, maximum_m=args.dmax_m)
    finally:
        writer.release()
    metadata.update({"source_zarr": str(args.depth_zarr), "source_dtype": str(depth.dtype),
                     "validity_source": args.valid_key or "finite_positive_depth_only",
                     "fps": float(fps), "width": depth.shape[2], "height": depth.shape[1],
                     "video": video.name, "preview": preview.name,
                     "note": "Reads stored depth; use inference cache mode to visualize pre-compression depth."})
    metadata["preview_compression"] = compress_preview_video(video, output_path=preview)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
