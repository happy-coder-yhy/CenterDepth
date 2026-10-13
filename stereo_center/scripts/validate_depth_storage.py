"""Repack and validate the S storage scheme against decoded source depth, not GT."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stereo_center.depth_sz3  # noqa: F401
from stereo_center.depth_zarr import DepthZarrWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.002)
    args = parser.parse_args()
    source_root = zarr.open_group(str(args.source), mode="r")
    source = source_root["depth"]
    if source.ndim != 3 or source.dtype != np.dtype("float32"):
        parser.error("source must be a (T,H,W) metric float32 depth array")
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "depth.zarr"
    digest = hashlib.sha256()
    elapsed = 0.0
    with DepthZarrWriter(path, *source.shape[1:], metadata=dict(source_root.attrs), alpha=args.alpha) as writer:
        for frame in source:
            digest.update(np.ascontiguousarray(frame, dtype="<f4").tobytes())
            start = time.perf_counter()
            writer.append(frame)
            elapsed += time.perf_counter() - start
    decoded = zarr.open_group(str(path), mode="r")["depth"]
    count = exact_count = mismatched = over = 0
    total = squares = maximum = max_ratio = read_seconds = 0.0
    for i in range(len(source)):
        original = source[i]
        start = time.perf_counter()
        result = decoded[i]
        read_seconds += time.perf_counter() - start
        valid = np.isfinite(original) & (original > 0) & (original <= 10)
        exact_count += int((~valid).sum())
        mismatched += int((original.view("<u4")[~valid] != result.view("<u4")[~valid]).sum())
        if not np.isfinite(result[valid]).all() or not (result[valid] > 0).all():
            raise ValueError("Invalid reconstructed normal depth")
        error = np.abs(result[valid].astype("f8") - original[valid])
        budget = args.alpha * np.maximum(1, original[valid].astype("f8"))
        count += error.size
        total += float(error.sum())
        squares += float(np.square(error).sum())
        maximum = max(maximum, float(error.max(initial=0)))
        max_ratio = max(max_ratio, float((error / budget).max(initial=0)))
        over += int((error > budget).sum())
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    report = {
        "source": str(args.source), "source_decoded_sha256": digest.hexdigest(),
        "reference": "decoded source prediction, NOT independent ground truth",
        "shape": list(source.shape), "storage": writer.storage_metadata,
        "bytes": size, "mib": size / 2**20,
        "normal_pixels": count, "exact_pixels": exact_count,
        "exact_bit_mismatches": mismatched, "pixels_over_bound": over,
        "mae_m": total / count if count else None,
        "rmse_m": (squares / count)**0.5 if count else None,
        "max_error_m": maximum, "max_error_budget_ratio": max_ratio,
        "write_seconds_excluding_source_read": elapsed,
        "read_seconds": read_seconds,
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if mismatched or over:
        raise SystemExit("S storage validation FAILED")


if __name__ == "__main__":
    main()
