"""Sequence-wide depth color scales, independent of depth storage compression."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import time

import numpy as np

from .visualize import colorize_depth_log


def _valid_values(depth, valid):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if depth.shape != valid.shape:
        raise ValueError("Depth and validity mask shapes must match")
    return depth[valid & np.isfinite(depth) & (depth > 0)]


def sequence_depth_scale(
    frame_source, *, percentile=99.9, step_m=0.5, minimum_m=1.0, maximum_m=20.0
):
    """Exact float32 percentile with two streaming passes and bounded histograms.

    frame_source must return a fresh, unchanged iterator of (depth, valid) pairs.
    Positive finite float32 bit patterns have the same order as their values.
    Count the upper 16 bits first, then the lower bits of the one or two buckets
    containing the required order statistics. No full-sequence sort/allocation.
    """
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be positive and finite")
    if not math.isfinite(minimum_m) or minimum_m <= 0:
        raise ValueError("minimum_m must be positive and finite")
    if not math.isfinite(maximum_m) or maximum_m < minimum_m:
        raise ValueError("maximum_m must be finite and at least minimum_m")
    counts = np.zeros(65536, dtype=np.int64)
    n_frames = n_pixels = 0
    for depth, valid in frame_source():
        values = _valid_values(depth, valid)
        counts += np.bincount(values.view(np.uint32) >> 16, minlength=65536)
        n_frames += 1
        n_pixels += np.asarray(depth).size
    if not n_frames:
        raise ValueError("Cannot select a color scale for an empty sequence")
    cumulative = counts.cumsum()
    n_valid = int(cumulative[-1])
    quantile = None
    if n_valid:
        rank = (n_valid - 1) * (percentile / 100)
        ranks = [math.floor(rank), math.ceil(rank)]
        buckets = [int(np.searchsorted(cumulative, r + 1)) for r in ranks]
        low_counts = {b: np.zeros(65536, dtype=np.int64) for b in buckets}
        second_frames = second_valid = 0
        for depth, valid in frame_source():
            bits = _valid_values(depth, valid).view(np.uint32)
            upper = bits >> 16
            for bucket, hist in low_counts.items():
                hist += np.bincount(bits[upper == bucket] & 65535, minlength=65536)
            second_frames += 1
            second_valid += bits.size
        if second_frames != n_frames or second_valid != n_valid:
            raise ValueError("Frame source changed between percentile passes")
        values_at_rank = []
        for r, b in zip(ranks, buckets):
            hist = low_counts[b]
            if hist.sum() != counts[b]:
                raise ValueError("Frame source changed between percentile passes")
            offset = int(cumulative[b - 1]) if b else 0
            low = int(np.searchsorted(hist.cumsum(), r - offset + 1))
            bits = np.array([(b << 16) | low], dtype=np.uint32)
            values_at_rank.append(float(bits.view(np.float32)[0]))
        a, b = values_at_rank
        fraction = rank - ranks[0]
        quantile = a + (b - a) * fraction
    upper = max(minimum_m, quantile if quantile is not None else minimum_m)
    uncapped_vmax = math.ceil(upper / step_m) * step_m
    vmax = min(maximum_m, uncapped_vmax)
    return {
        "mode": "sequence-p999", "mapping": "log", "gamma": 0.6,
        "colormap": "JET", "dmin_m": 0.3, "dmax_m": vmax,
        "percentile": percentile, "percentile_depth_m": quantile,
        "percentile_method": "exact_float32_two_pass_radix_linear_interpolation",
        "round_up_step_m": step_m, "minimum_range_m": minimum_m,
        "maximum_range_m": maximum_m, "uncapped_dmax_m": uncapped_vmax,
        "range_capped": uncapped_vmax > maximum_m,
        "n_frames": n_frames, "n_pixels": int(n_pixels), "valid_pixels": n_valid,
        "fallback": "no_valid_positive_depth" if not n_valid else None,
        "fixed_across_frames": True,
    }


def render_adaptive_frames(frame_source, write_frame, *, maximum_m=20.0):
    """Call write_frame(index, BGR) with a single global scale for the sequence."""
    start = time.perf_counter()
    metadata = sequence_depth_scale(frame_source, maximum_m=maximum_m)
    metadata["range_selection_seconds"] = time.perf_counter() - start
    saturated = below_minimum = rendered = valid_count = 0
    start = time.perf_counter()
    for index, (depth, valid) in enumerate(frame_source()):
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
        saturated += int(np.count_nonzero(valid & (depth > metadata["dmax_m"])))
        below_minimum += int(np.count_nonzero(valid & (depth < metadata["dmin_m"])))
        valid_count += int(np.count_nonzero(valid))
        write_frame(index, colorize_depth_log(
            depth, valid, d_min=metadata["dmin_m"], d_max=metadata["dmax_m"]
        ))
        rendered += 1
    if rendered != metadata["n_frames"] or valid_count != metadata["valid_pixels"]:
        raise ValueError("Frame source changed between selection and rendering")
    metadata["render_seconds"] = time.perf_counter() - start
    metadata["saturated_pixels"] = saturated
    metadata["saturated_fraction_of_valid"] = saturated / valid_count if valid_count else 0.0
    metadata["below_minimum_pixels"] = below_minimum
    metadata["below_minimum_fraction_of_valid"] = below_minimum / valid_count if valid_count else 0.0
    return metadata


class DepthVisualizationCache:
    """Lossless temporary depth/mask spool; never reads or changes the output Zarr."""

    def __init__(self, directory):
        self._temp = tempfile.TemporaryDirectory(prefix=".depth-color-", dir=directory)
        self.path = Path(self._temp.name)
        self.frame_indices = []

    def append(self, depth, valid, frame_index):
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if depth.ndim != 2 or depth.shape != valid.shape:
            raise ValueError("Expected matching 2D depth and mask")
        np.savez(self.path / f"{len(self.frame_indices):08d}.npz", depth=depth, valid=valid)
        self.frame_indices.append(int(frame_index))

    def frames(self):
        for i in range(len(self.frame_indices)):
            with np.load(self.path / f"{i:08d}.npz", allow_pickle=False) as data:
                yield data["depth"], data["valid"]

    def close(self):
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
