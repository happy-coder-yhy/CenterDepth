"""Incremental Zarr storage for metric depth video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


DEPTH_ABSOLUTE_ERROR_METER = 0.001
DEPTH_CHUNK_FRAMES = 32


class DepthZarrWriter:
    """Buffer metric frames and write each spatiotemporal chunk exactly once.

    Call close() to persist the final partial chunk. Readers see only flushed
    frames while the writer is open; n_frames includes buffered frames.
    """

    def __init__(
        self,
        path: str | Path,
        height: int,
        width: int,
        metadata: Mapping[str, object] | None = None,
        *,
        chunk_frames: int = DEPTH_CHUNK_FRAMES,
    ) -> None:
        try:
            import zarr
            import pysz  # noqa: F401
            from .depth_sz3 import DepthSZ3
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Zarr output requires `zarr<3`, `numcodecs` and `pysz==1.0.3`; "
                "install them in the runtime environment"
            ) from exc

        if height < 1 or width < 1:
            raise ValueError(f"Depth Zarr dimensions must be positive, got {height}x{width}")
        if (
            isinstance(chunk_frames, bool)
            or not isinstance(chunk_frames, (int, np.integer))
            or chunk_frames < 1
        ):
            raise ValueError("chunk_frames must be a positive integer")

        self.path = Path(path)
        self.height = int(height)
        self.width = int(width)
        self.chunk_frames = int(chunk_frames)
        self._buffer = np.empty((self.chunk_frames, self.height, self.width), dtype=np.float32)
        self._buffer_count = 0
        self._written_frames = 0
        self._closed = False
        self._root = zarr.open_group(str(self.path), mode="w")
        self._depth = self._root.create_dataset(
            "depth",
            shape=(0, self.height, self.width),
            chunks=(self.chunk_frames, self.height, self.width),
            dtype="f4",
            compressor=DepthSZ3(
                shape=(self.chunk_frames, self.height, self.width),
                tolerance=DEPTH_ABSOLUTE_ERROR_METER,
            ),
        )
        attrs = dict(metadata or {})
        attrs.update(
            {
                "n_frames": 0,
                "height": self.height,
                "width": self.width,
                "dtype": "float32",
                "depth_unit": "meter",
                "storage_encoding": "sz3",
                "storage_dtype": "float32",
                "error_bound_mode": "ABS",
                "absolute_error_tolerance_m": DEPTH_ABSOLUTE_ERROR_METER,
                "codec_id": DepthSZ3.codec_id,
                "reader_requirement": "pysz and import stereo_center.depth_sz3",
                "invalid_depth_value": 0.0,
                "compression_axes": ["time", "height", "width"],
                "temporal_chunk_frames": self.chunk_frames,
                "partial_chunk_padding": "zero; excluded from array shape",
            }
        )
        self._root.attrs.update(attrs)

    @property
    def n_frames(self) -> int:
        return self._written_frames + self._buffer_count

    def append(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed DepthZarrWriter")
        depth = np.array(frame, dtype=np.float32, copy=True)
        expected = (self.height, self.width)
        if depth.shape != expected:
            raise ValueError(f"Depth frame must have shape {expected}, got {depth.shape}")
        np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(depth, 0.0, out=depth)
        self._buffer[self._buffer_count] = depth
        self._buffer_count += 1
        if self._buffer_count == self.chunk_frames:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._buffer_count:
            return
        stop = self._written_frames + self._buffer_count
        self._depth.resize((stop, self.height, self.width))
        # Never update a previously compressed lossy chunk frame by frame.
        self._depth[self._written_frames:stop] = self._buffer[:self._buffer_count]
        self._written_frames = stop
        self._buffer_count = 0
        self._root.attrs["n_frames"] = stop

    def close(self) -> None:
        if self._closed:
            return
        self._flush_chunk()
        self._closed = True

    def __enter__(self) -> "DepthZarrWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
