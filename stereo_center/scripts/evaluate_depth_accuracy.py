"""Compare uncompressed and reconstructed depth to an independent metric reference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stereo_center.depth_sz3  # noqa: F401
from stereo_center.depth_accuracy import DepthAccuracy


class NpyDepthFrames:
    """Read the lossless per-frame outputs from --save-depth-npy 1."""

    def __init__(self, path):
        self.paths = sorted(path.glob("depth_*.npy"), key=lambda p: int(p.stem[6:]))
        if not self.paths:
            raise ValueError(f"No depth_*.npy frames in {path}")
        indices = [int(p.stem[6:]) for p in self.paths]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("Missing or duplicate NPY frame indices")
        first = np.load(self.paths[0], mmap_mode="r", allow_pickle=False)
        if first.ndim != 2:
            raise ValueError("Per-frame depth must be (H,W)")
        self.shape = (len(self.paths), *first.shape)

    def __getitem__(self, index):
        frame = np.load(self.paths[index], allow_pickle=False)
        if frame.shape != self.shape[1:]:
            raise ValueError("Inconsistent NPY frame shape")
        return frame

    def __len__(self):
        return self.shape[0]


def open_depth(path):
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.is_dir() and not (path / ".zgroup").exists():
        return NpyDepthFrames(path)
    return zarr.open_group(str(path), mode="r")["depth"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reconstructed", required=True, type=Path)
    parser.add_argument("--reference-kind", required=True, choices=["ground_truth", "sensor_reference"])
    parser.add_argument("--reference-description", required=True)
    parser.add_argument("--reference-uncertainty-m", type=float)
    parser.add_argument("--aligned-metric-z-confirmed", action="store_true", required=True,
                        help="All arrays are aligned in time and pixels, in meters, optical-axis Z (not range)")
    parser.add_argument("--mask", type=Path, help="Optional reference-only boolean (T,H,W) NPY mask")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.reference.resolve() in (args.prediction.resolve(), args.reconstructed.resolve()):
        parser.error("Reference must be independent of predictions")
    if args.reference_uncertainty_m is not None and (
        not np.isfinite(args.reference_uncertainty_m) or args.reference_uncertainty_m < 0
    ):
        parser.error("Reference uncertainty must be finite and nonnegative")
    if args.output.exists():
        parser.error("Output exists; refusing to overwrite")
    prediction, reference, reconstructed = [open_depth(p) for p in
                                             (args.prediction, args.reference, args.reconstructed)]
    if prediction.shape != reference.shape or reconstructed.shape != reference.shape or len(reference.shape) != 3:
        parser.error("Inputs must have identical (T,H,W) shapes; no implicit registration/resizing")
    mask = np.load(args.mask, mmap_mode="r", allow_pickle=False) if args.mask else None
    if mask is not None and (mask.shape != reference.shape or mask.dtype != np.dtype(bool)):
        parser.error("Mask must be boolean and match (T,H,W)")
    base, stored, base_common, stored_common = [DepthAccuracy() for _ in range(4)]
    for i in range(len(reference)):
        gt, raw, restored = reference[i], prediction[i], reconstructed[i]
        valid = mask[i] if mask is not None else None
        base.update(raw, gt, valid)
        stored.update(restored, gt, valid)
        common = np.isfinite(raw) & (raw > 0) & np.isfinite(restored) & (restored > 0)
        if valid is not None:
            common &= valid
        base_common.update(raw, gt, common)
        stored_common.update(restored, gt, common)
    before, after = base_common.result(), stored_common.result()
    result = {
        "reference_kind": args.reference_kind,
        "reference_description": args.reference_description,
        "reference_uncertainty_m": args.reference_uncertainty_m,
        "inputs": {k: str(getattr(args, k)) for k in ("prediction", "reference", "reconstructed", "mask")},
        "units": "meters, optical-axis Z", "shape": list(reference.shape),
        "prediction": base.result(), "reconstructed": stored.result(),
        "common_support_prediction": before, "common_support_reconstructed": after,
        "common_support_delta": {
            region: {key: after[region][key] - row[key] if row[key] is not None else None
                     for key in ("mae_m", "rmse_m", "bias_m", "abs_rel")}
            for region, row in before.items()
        },
        "interpretation": "No automatic precision selection: set downstream tolerances and inspect coverage and reference uncertainty.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
