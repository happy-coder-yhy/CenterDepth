"""Compare bounded-error SZ3 Zarr stores against original metric depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_center.depth_sz3 import DepthSZ3


def file_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-mp4", type=Path, required=True)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.0005, 0.005, 0.01, 0.03, 0.05])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_group = zarr.open_group(str(args.source), mode="r")
    source = source_group["depth"]
    chunks = (1, *source.shape[1:])
    report = {
        "source": str(args.source), "shape": list(source.shape),
        "source_bytes": file_bytes(args.source),
        "reference_mp4_bytes": args.reference_mp4.stat().st_size,
        "error_reference": "Original FFS float32 output, not ground truth",
        "candidates": [],
    }
    for tolerance in args.tolerances:
        path = args.output / f"sz3_abs_{tolerance:g}m.zarr"
        root = zarr.open_group(str(path), mode="w")
        root.attrs.update(dict(source_group.attrs))
        root.attrs.update({"compression": "SZ3", "absolute_error_tolerance_m": tolerance,
                           "reader_requirement": "pysz plus import stereo_center.depth_sz3"})
        dst = root.create_dataset("depth", shape=source.shape, chunks=chunks,
                                  dtype="<f4", compressor=DepthSZ3(chunks, tolerance))
        write_seconds = 0.0
        for i in range(len(source)):
            frame = source[i]
            start = time.perf_counter()
            dst[i] = frame
            write_seconds += time.perf_counter() - start
        # Reopen through the codec registry, then validate every stored pixel.
        dst = zarr.open_group(str(path), mode="r")["depth"]
        count = 0
        sum_abs = sum_sq = maximum = 0.0
        over_bound = nonfinite = negative = over_1cm = 0
        hist = np.zeros(10001, dtype=np.int64)
        read_seconds = 0.0
        for i in range(len(source)):
            frame = source[i].astype(np.float64)
            start = time.perf_counter()
            decoded = dst[i].astype(np.float64)
            read_seconds += time.perf_counter() - start
            error = np.abs(decoded - frame)
            if not np.isfinite(error).all():
                raise ValueError(f"Nonfinite reconstruction error: {path}, frame {i}")
            count += error.size
            sum_abs += float(error.sum())
            sum_sq += float(np.square(error).sum())
            maximum = max(maximum, float(error.max()))
            over_bound += int((error > tolerance).sum())
            over_1cm += int((error > 0.01).sum())
            nonfinite += int((~np.isfinite(decoded)).sum())
            negative += int((decoded < 0).sum())
            bins = np.minimum((error / tolerance * 10000).astype(np.int64), 10000)
            hist += np.bincount(bins.ravel(), minlength=10001)
        size = file_bytes(path)
        p99_index = int(np.searchsorted(hist.cumsum(), int(np.ceil(count * 0.99))))
        result = {
            "path": str(path), "tolerance_m": tolerance, "bytes": size,
            "mib": size / 2**20, "smaller_than_mp4": size < report["reference_mp4_bytes"],
            "mae_m": sum_abs / count, "rmse_m": (sum_sq / count)**0.5,
            "max_error_m": maximum,
            "p99_upper_m": (p99_index + 1) * tolerance / 10000 if p99_index < 10000 else None,
            "pixels_over_tolerance": over_bound, "pixels_over_1cm_pct": 100 * over_1cm / count,
            "nonfinite_pixels": nonfinite, "negative_pixels": negative,
            "write_seconds": write_seconds, "read_seconds": read_seconds,
            "all_pixels": count,
        }
        report["candidates"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
