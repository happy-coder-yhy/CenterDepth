"""Incremental Zarr storage for metric depth video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


class DepthZarrWriter:
    """Append a metric depth sequence to a Zarr v2 group."""

    def __init__(
        self,
        path: str | Path,
        height: int,
        width: int,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Zarr output requires `zarr<3` and `numcodecs`; install them in the runtime environment"
            ) from exc

        if height < 1 or width < 1:
            raise ValueError(f"Depth Zarr dimensions must be positive, got {height}x{width}")

        self.path = Path(path)
        self.height = int(height)
        self.width = int(width)
        self._closed = False
        self._root = zarr.open_group(str(self.path), mode="w")
        self._depth = self._root.create_dataset(
            "depth",
            shape=(0, self.height, self.width),
            chunks=(1, self.height, self.width),
            dtype="f4",
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        )
        attrs = dict(metadata or {})
        attrs.update(
            {
                "n_frames": 0,
                "height": self.height,
                "width": self.width,
                "dtype": "float32",
                "depth_unit": "meter",
            }
        )
        self._root.attrs.update(attrs)

    @property
    def n_frames(self) -> int:
        return int(self._depth.shape[0])

    def append(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed DepthZarrWriter")
        depth = np.asarray(frame, dtype=np.float32)
        expected = (self.height, self.width)
        if depth.shape != expected:
            raise ValueError(f"Depth frame must have shape {expected}, got {depth.shape}")
        next_index = self.n_frames
        self._depth.resize((next_index + 1, self.height, self.width))
        self._depth[next_index] = depth
        self._root.attrs["n_frames"] = next_index + 1

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "DepthZarrWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
