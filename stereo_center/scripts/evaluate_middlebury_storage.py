"""Evaluate the deployed FFS engine and S storage on Middlebury trainingQ.

This is a native-Q engineering evaluation, not the official full-res leaderboard.
No training, resizing, rectification, GT fitting, or temporal filtering is done.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_center.depth_accuracy import DepthAccuracy
from stereo_center.depth_zarr import DepthZarrWriter
from stereo_center.visualize import colorize_depth_log


def read_pfm(path):
    with Path(path).open("rb") as handle:
        def line():
            while True:
                value = handle.readline()
                if not value:
                    raise ValueError("Truncated PFM header")
                if value.strip() and not value.startswith(b"#"):
                    return value.strip()
        if line() != b"Pf":
            raise ValueError("Expected single-channel disparity PFM")
        width, height = map(int, line().split())
        scale = float(line())
        if min(width, height) <= 0 or not np.isfinite(scale) or scale == 0:
            raise ValueError("Expected positive shape and finite nonzero PFM scale")
        # Match Middlebury's ImageIO.cpp: magnitude is not a disparity multiplier.
        data = np.frombuffer(handle.read(), dtype="<f4" if scale < 0 else ">f4")
        if data.size != width * height:
            raise ValueError("PFM payload size mismatch")
        return np.flipud(data.reshape(height, width)).astype("f4", copy=True)


def read_calibration(path):
    fields = dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)
    camera = np.fromstring(fields["cam0"].strip("[]").replace(";", " "), sep=" ").reshape(3, 3)
    result = {"fx": float(camera[0, 0]), "baseline_mm": float(fields["baseline"]),
              "doffs": float(fields["doffs"]), "width": int(fields["width"]),
              "height": int(fields["height"]), "isint": int(fields.get("isint", 0))}
    if result["fx"] <= 0 or result["baseline_mm"] <= 0:
        raise ValueError("Invalid focal length or baseline")
    return result


def metric_depth(disparity, calib):
    disparity = np.asarray(disparity, dtype=np.float64)
    denominator = disparity + calib["doffs"]
    valid = np.isfinite(disparity) & (disparity >= 0) & (denominator > 0)
    depth = np.zeros(disparity.shape, dtype="f4")
    depth[valid] = calib["baseline_mm"] * calib["fx"] / denominator[valid] / 1000
    return depth


def disparity_metrics(pred, gt, mask):
    valid = mask & np.isfinite(gt) & (gt >= 0)
    supported = valid & np.isfinite(pred) & (pred >= 0)
    error = np.abs(pred[supported].astype("f8") - gt[supported])
    return {"gt_pixels": int(valid.sum()), "evaluated_pixels": error.size,
            "epe_px": float(error.mean()) if error.size else None,
            **{f"bad_{x:g}px_fraction": float((error > x).mean()) if error.size else None
               for x in (0.5, 1, 2, 3)}}, error


def storage_audit(raw, decoded, alpha):
    normal = np.isfinite(raw) & (raw > 0) & (raw <= 10)
    error = np.abs(raw[normal].astype("f8") - decoded[normal])
    budget = alpha * np.maximum(1, raw[normal].astype("f8"))
    mismatch = int((raw.view("u4")[~normal] != decoded.view("u4")[~normal]).sum())
    over = int((error > budget).sum())
    if mismatch or over or not np.isfinite(decoded[normal]).all() or not (decoded[normal] > 0).all():
        raise ValueError("Storage reconstruction failed its error contract")
    return {"mae_m": float(error.mean()) if error.size else None,
            "max_error_m": float(error.max(initial=0)), "pixels_over_bound": over,
            "exact_bit_mismatches": mismatch}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_summary(pred, gt, mask):
    accumulator = DepthAccuracy()
    accumulator.update(pred, gt, mask)
    result = accumulator.result()
    supported = mask & np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    error = np.abs(pred[supported].astype("f8") - gt[supported])
    for q in (50, 95, 99):
        result["all"][f"p{q}_m"] = float(np.percentile(error, q)) if error.size else None
    return result, error


def comparison_image(name, rgb, gt, raw, stored, valid):
    samples = gt[valid]
    maximum = max(1, min(20, np.ceil(np.percentile(samples, 99.9) * 2) / 2))
    colors = [colorize_depth_log(z, valid & (z > 0) & np.isfinite(z), 0.3, maximum)
              for z in (gt, raw, stored)]
    errors = [np.abs(raw.astype("f8") - gt), np.abs(stored.astype("f8") - raw)]
    heat = []
    for err, cap in zip(errors, (0.2, 0.01)):
        safe = np.where(valid & np.isfinite(err), err, 0)
        colored = cv2.applyColorMap(np.uint8(np.clip(safe / cap, 0, 1) * 255), cv2.COLORMAP_INFERNO)
        colored[~valid] = 0
        heat.append(colored)
    labels = [f"{name}: RGB (static scene)", f"GT: 0.3-{maximum:g}m", "FFS: same depth scale",
              "S2 decoded: same depth scale", "FFS vs GT: 0-200mm", "S2 vs FFS: 0-10mm"]
    tiles = []
    for content, label in zip([rgb, *colors, *heat], labels):
        tile = np.full((360, 480, 3), 245, np.uint8)
        scale = min(480 / content.shape[1], 320 / content.shape[0])
        w, h = round(content.shape[1] * scale), round(content.shape[0] * scale)
        x, y = (480 - w) // 2, 40 + (320 - h) // 2
        tile[y:y+h, x:x+w] = cv2.resize(content, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Official trainingQ directory")
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--ffs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    scenes = sorted(p.parent for p in args.data.glob("*/disp0GT.pfm"))
    if len(scenes) != 15:
        parser.error(f"Expected all 15 trainingQ scenes, found {len(scenes)}")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from stereo_center.ffs_tensorrt_inference import load_ffs_tensorrt, run_stereo_matching_batch
    model = load_ffs_tensorrt(args.engine_dir, args.ffs_root)
    tiers = {"S1": 0.001, "S2": 0.002, "S3": 0.003}
    pooled = {name: {mask: DepthAccuracy() for mask in ("ALL", "NOC")} for name in ("raw", *tiers)}
    error_samples = {name: {mask: [] for mask in ("ALL", "NOC")} for name in pooled}
    disp_samples = {mask: [] for mask in ("ALL", "NOC")}
    sizes = {name: 0 for name in tiers}
    report = {"evaluation": "Middlebury trainingQ native-resolution; not official leaderboard",
              "inference": "existing TensorRT engine, no training or GT-dependent preprocessing",
              "engine_sha256": {p.name: sha256(p) for p in args.engine_dir.glob("*.engine")},
              "engine_config": (args.engine_dir / "onnx.yaml").read_text(),
              "data": str(args.data), "scenes": [], "alpha": tiers}
    video = cv2.VideoWriter(str(args.output / "comparison_static_scenes.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 5, (1440, 720))
    if not video.isOpened():
        raise RuntimeError("Cannot open comparison video")
    try:
        for scene in scenes:
            calib = read_calibration(scene / "calib.txt")
            gt_disp = read_pfm(scene / "disp0GT.pfm")
            left, right = [cv2.imread(str(scene / filename)) for filename in ("im0.png", "im1.png")]
            shape = (calib["height"], calib["width"])
            if left is None or right is None or left.shape != right.shape or left.shape[:2] != shape or gt_disp.shape != shape:
                raise ValueError(f"Inconsistent image/calibration shapes in {scene}")
            gt = metric_depth(gt_disp, calib)
            gt_valid = gt > 0
            noc = cv2.imread(str(scene / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
            if noc is None or noc.shape != shape:
                raise ValueError("Missing or mismatched NOC mask")
            masks = {"ALL": gt_valid, "NOC": gt_valid & (noc == 255)}
            tensors = [torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None]
                       for im in (left, right)]
            if not report["scenes"]:
                run_stereo_matching_batch(model, *tensors)  # Warm up the fixed engine and Triton.
            started = time.perf_counter()
            prediction, _, _, engine_seconds = run_stereo_matching_batch(model, *tensors)
            inference_seconds = time.perf_counter() - started
            disp = prediction[0].numpy()
            raw = metric_depth(disp, calib)
            destination = args.output / scene.name
            destination.mkdir()
            np.save(destination / "prediction_disparity.npy", disp)
            np.save(destination / "prediction_depth_m.npy", raw)
            row = {"scene": scene.name, "shape": list(shape), "calibration": calib,
                   "input_sha256": {name: sha256(scene / name) for name in
                                    ("im0.png", "im1.png", "disp0GT.pfm", "mask0nocc.png", "calib.txt")},
                   "engine_call_seconds": engine_seconds, "inference_seconds": inference_seconds,
                   "gt_disparity_ge_engine_max_pixels": int((gt_valid & (gt_disp >= model.runner.args.max_disp)).sum()),
                   "depth": {}, "disparity": {}, "storage": {}}
            variants = {"raw": raw}
            for name, alpha in tiers.items():
                path = destination / f"{name}.zarr"
                start = time.perf_counter()
                with DepthZarrWriter(path, *shape, alpha=alpha) as writer:
                    writer.append(raw)
                encoded_seconds = time.perf_counter() - start
                decoded = zarr.open_group(str(path), mode="r")["depth"][0]
                variants[name] = decoded
                size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                sizes[name] += size
                row["storage"][name] = {"bytes": size, "encode_seconds": encoded_seconds,
                                         **storage_audit(raw, decoded, alpha)}
            for mask_name, mask in masks.items():
                row["disparity"][mask_name], errors = disparity_metrics(disp, gt_disp, mask)
                disp_samples[mask_name].append(errors)
                for name, depth in variants.items():
                    row["depth"].setdefault(name, {})
                    row["depth"][name][mask_name], errors = depth_summary(depth, gt, mask)
                    pooled[name][mask_name].update(depth, gt, mask)
                    error_samples[name][mask_name].append(errors)
            image = comparison_image(scene.name, left, gt, raw, variants["S2"], gt_valid)
            if not cv2.imwrite(str(destination / "comparison.png"), image):
                raise RuntimeError("Could not write comparison image")
            for _ in range(15):
                video.write(image)
            (destination / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            report["scenes"].append(row)
            print(json.dumps({"scene": scene.name, "epe": row["disparity"]["NOC"]["epe_px"],
                              "depth_mae_m": row["depth"]["raw"]["NOC"]["all"]["mae_m"]}), flush=True)
    finally:
        video.release()
    report["storage_bytes"] = sizes
    report["pooled_depth"] = {name: {mask: accumulator.result() for mask, accumulator in groups.items()}
                              for name, groups in pooled.items()}
    for name, groups in error_samples.items():
        for mask, arrays in groups.items():
            errors = np.concatenate(arrays)
            report["pooled_depth"][name][mask]["all"].update(
                {f"p{q}_m": float(np.percentile(errors, q)) for q in (50, 95, 99)})
    report["pooled_disparity"] = {}
    for mask, arrays in disp_samples.items():
        errors = np.concatenate(arrays)
        report["pooled_disparity"][mask] = {"pixels": len(errors), "epe_px": float(errors.mean()),
            **{f"bad_{q:g}px_fraction": float((errors > q).mean()) for q in (0.5, 1, 2, 3)}}
    report["macro_depth_mae_m"] = {name: {mask: float(np.mean(
        [r["depth"][name][mask]["all"]["mae_m"] for r in report["scenes"]])) for mask in ("ALL", "NOC")}
        for name in pooled}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"storage_bytes": sizes, "pooled_disparity": report["pooled_disparity"]}), flush=True)


if __name__ == "__main__":
    main()
