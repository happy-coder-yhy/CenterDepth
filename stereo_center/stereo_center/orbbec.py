"""Orbbec split-stereo recording helpers.

Orbbec recordings store the two camera streams separately and may drop a
different number of frames on each side.  Pair frames by their device PTS,
not by their ordinal frame number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def pts_sidecar_path(video_path: str | Path) -> Path:
    """Return the recorder's conventional ``*_pts.csv`` sidecar path."""
    path = Path(video_path)
    return path.with_name(f"{path.stem}_pts.csv")


def load_pts_us(path: str | Path) -> np.ndarray:
    """Load a recorder PTS CSV containing a ``timestamp_us`` header."""
    values = np.loadtxt(Path(path), delimiter=",", skiprows=1, dtype=np.int64)
    return np.atleast_1d(values)


def pts_metadata_mismatch(
    left_pts_count: int,
    right_pts_count: int,
    left_container_frames: int,
    right_container_frames: int,
) -> bool:
    """Whether container-reported frame counts disagree with PTS sidecars."""
    return (left_pts_count, right_pts_count) != (
        left_container_frames,
        right_container_frames,
    )


def forward_decode_read_count(
    previous_index: int | None, target_index: int
) -> int | None:
    """Return sequential reads needed for ``target_index``, or ``None`` to seek."""
    if previous_index is None or target_index < previous_index:
        return None
    return target_index - previous_index


def match_left_to_right_pts(
    left_pts_us: np.ndarray,
    right_pts_us: np.ndarray,
    max_delta_us: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Match every left timestamp to its nearest right timestamp in tolerance."""
    left = np.asarray(left_pts_us, dtype=np.int64)
    right = np.asarray(right_pts_us, dtype=np.int64)
    if left.ndim != 1 or right.ndim != 1 or len(left) == 0 or len(right) == 0:
        raise ValueError("left/right PTS must be non-empty one-dimensional arrays")
    if max_delta_us < 0:
        raise ValueError("max_delta_us must be non-negative")

    insertion = np.searchsorted(right, left, side="left")
    after = np.clip(insertion, 0, len(right) - 1)
    before = np.clip(insertion - 1, 0, len(right) - 1)
    after_delta = np.abs(right[after] - left)
    before_delta = np.abs(right[before] - left)
    right_indices = np.where(before_delta <= after_delta, before, after)
    deltas = np.abs(right[right_indices] - left)
    valid = deltas <= max_delta_us
    pairs = [(int(left_idx), int(right_idx)) for left_idx, right_idx in zip(
        np.flatnonzero(valid), right_indices[valid]
    )]
    return pairs, deltas[valid]
