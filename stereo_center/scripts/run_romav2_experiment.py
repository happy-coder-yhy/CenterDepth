#!/usr/bin/env python
"""Reproducible staged RoMa/SEA temporal correspondence and FFS experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from stereo_center import calib
from stereo_center.orbbec import load_pts_us, match_left_to_right_pts, pts_sidecar_path
from stereo_center.temporal_stereo import backward_warp, temporal_alignment_mask
from stereo_center.temporal_geometry import estimate_relative_pose_pnp, reproject_depth_zbuffer
from stereo_center.visualize import colorize_depth_log
from stereo_center.flow_metrics import region_flow_metrics


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].cuda().float()


def scalar_image(array):
    return torch.from_numpy(np.array(array))[None, None].cuda().float()


def samples_summary(values):
    data = np.concatenate(values) if values else np.array([])
    if not data.size:
        return {"count": 0}
    return {"count": int(data.size), "mean": float(data.mean(dtype=np.float64)),
            "median": float(np.median(data)), "p95": float(np.quantile(data, .95)),
            "p99": float(np.quantile(data, .99))}


def video_writer(path, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (800, 650))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {path}")
    return writer


def draw_depth(writer, depth):
    writer.write(colorize_depth_log(depth, np.isfinite(depth) & (depth > 0)))


class SequentialReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot read {path}")
        self.index, self.frame = -1, None

    def read(self, target):
        if target < self.index:
            raise ValueError("Expected nondecreasing PTS indices")
        while self.index < target:
            ok, self.frame = self.cap.read()
            if not ok:
                raise RuntimeError(f"Incomplete decode at {target}")
            self.index += 1
        return self.frame


def prepare(args):
    from stereo_center.ffs_inference import load_ffs
    from evaluate_ffs_temporal_warmstart import native_ffs
    start = time.perf_counter()
    prefix = args.root / "dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031"
    stem = prefix / prefix.name
    left_path, right_path = Path(str(stem)+"_camera_left_part0001.mp4"), Path(str(stem)+"_camera_right_part0001.mp4")
    calibration = Path(str(stem)+"_calibration_camera.yaml")
    rect = calib.compute_rectification_maps(calib.load_orbbec_calibration(calibration), (800, 650))
    lp, rp = load_pts_us(pts_sidecar_path(left_path)), load_pts_us(pts_sidecar_path(right_path))
    pairs, delta = match_left_to_right_pts(lp, rp, 1000)
    if len(pairs) != 973:
        raise RuntimeError(f"Expected 973 pairs, got {len(pairs)}")
    if args.limit:
        pairs = pairs[:args.limit]
    args.run.mkdir(parents=True, exist_ok=False)
    n = len(pairs)
    rgb = np.lib.format.open_memmap(args.run/"rgb.npy", mode="w+", dtype="uint8", shape=(n, 650, 800, 3))
    right = np.lib.format.open_memmap(args.run/"right.npy", mode="w+", dtype="uint8", shape=rgb.shape)
    disp = np.lib.format.open_memmap(args.run/"raw_disp.npy", mode="w+", dtype="float32", shape=(n, 650, 800))
    roi = ((rect["mapsL"][0] >= 0) & (rect["mapsL"][0] <= 1599)
           & (rect["mapsL"][1] >= 0) & (rect["mapsL"][1] <= 1299))
    np.save(args.run/"roi.npy", roi)
    left_reader, right_reader = SequentialReader(left_path), SequentialReader(right_path)
    fps = left_reader.cap.get(cv2.CAP_PROP_FPS)
    writer = video_writer(args.run/"raw_depth.mp4", fps)
    ffs = load_ffs(weights_dir=args.root/"CenterDepth/weights/fast_foundation_stereo/23-36-37",
                   ffs_root=args.root/"Fast-FoundationStereo", valid_iters=5)
    inference = []
    try:
        for t, (li, ri) in enumerate(pairs):
            a, b = calib.rectify_pair(left_reader.read(li), right_reader.read(ri), rect)
            rgb[t], right[t] = cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
            d, elapsed = native_ffs(ffs, tensor(rgb[t]), tensor(right[t]), None)
            disp[t] = d[0, 0].cpu().numpy()
            inference.append(elapsed)
            draw_depth(writer, rect["fx"]*rect["baseline"]/disp[t].clip(.001))
            if (t+1) % 100 == 0:
                print(f"prepare {t+1}/{n}", flush=True)
    finally:
        writer.release(); left_reader.cap.release(); right_reader.cap.release()
        rgb.flush(); right.flush(); disp.flush()
    manifest = {"n": n, "fps": fps, "fxB": rect["fx"]*rect["baseline"],
                "pairs": pairs, "left_pts_us": [int(lp[i]) for i, _ in pairs],
                "max_stereo_offset_us": int(delta.max()), "calibration": str(calibration),
                "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                "source": [str(left_path), str(right_path)],
                "torch": torch.__version__, "raw_forward_seconds": inference,
                "wall_seconds": time.perf_counter()-start}
    manifest["pair_hash"] = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    save_json(args.run/"manifest.json", manifest)


def load_flow_model(args):
    if args.backend == "sea":
        from stereo_center.sea_raft_flow import load_sea_raft, flow_between
        model = load_sea_raft(args.root/"models/sea_raft/spring540x960-M/model.safetensors",
                              args.root/"third_party/SEA-RAFT", args.root/"third_party/SEA-RAFT/config/eval/spring-M.json")
        def pair(a, b):
            f, _ = flow_between(model, a, b, iters=4, input_scale=.5)
            r, _ = flow_between(model, b, a, iters=4, input_scale=.5)
            return f, r, torch.ones_like(f[:, :1]), torch.ones_like(r[:, :1])
        return pair
    from stereo_center.romav2_flow import RoMaFlow
    return RoMaFlow(args.setting).pair


def flow_name(args):
    return "sea" if args.backend == "sea" else "roma_"+args.setting


def estimate_flow(args):
    start = time.perf_counter()
    meta = json.loads((args.run/"manifest.json").read_text())
    rgb = np.load(args.run/"rgb.npy", mmap_mode="r")
    name = flow_name(args)
    pair = load_flow_model(args)
    load_seconds = time.perf_counter()-start
    for _ in range(3):
        pair(tensor(rgb[0]), tensor(rgb[1]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    flow = np.lib.format.open_memmap(args.run/f"{name}_flow.npy", mode="w+", dtype="float32",
                                    shape=(meta["n"]-1, 2, 2, 650, 800))
    overlap = np.lib.format.open_memmap(args.run/f"{name}_overlap.npy", mode="w+", dtype="float32",
                                       shape=(meta["n"]-1, 2, 650, 800))
    times = []
    try:
        for t in range(1, meta["n"]):
            a, b = tensor(rgb[t-1]), tensor(rgb[t])
            torch.cuda.synchronize(); tick = time.perf_counter()
            f, r, fa, ra = pair(a, b)
            torch.cuda.synchronize(); times.append(time.perf_counter()-tick)
            flow[t-1] = torch.stack((f[0], r[0])).cpu().numpy()
            overlap[t-1] = torch.stack((fa[0, 0], ra[0, 0])).cpu().numpy()
            if t % 100 == 0:
                print(f"{name} flow {t}/{meta['n']-1}", flush=True)
    finally:
        flow.flush(); overlap.flush()
    save_json(args.run/f"{name}_flow_summary.json",
              {"pair_hash": meta["pair_hash"], "backend": name, "torch": torch.__version__,
               "pair_seconds": times, "load_seconds": load_seconds,
               "wall_seconds": time.perf_counter()-start,
               "peak_allocated_GiB": torch.cuda.max_memory_allocated()/2**30,
               "peak_reserved_GiB": torch.cuda.max_memory_reserved()/2**30})


def synthetic(args):
    rgb = np.load(args.run/"rgb.npy", mmap_mode="r")
    roi = np.load(args.run/"roi.npy")
    pair = load_flow_model(args)
    y, x = np.mgrid[:650, :800].astype(np.float32)
    records = []
    for index in np.linspace(0, len(rgb)-1, 8, dtype=int):
        a = np.array(rgb[index])
        for dx, dy in ((0, 0), (1, .5), (5, -3), (15, 10), (-25, -12), (60, 20)):
            b = cv2.remap(a, x-dx, y-dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # Known occluder in B; GT remains the underlying translating surface.
            cover = np.zeros((650, 800), np.uint8)
            if dx or dy:
                cover[240:340, 320:440] = 1
                b[240:340, 320:440] = np.roll(a[240:340, 320:440], 40, axis=1)
            tx, ty = x+dx, y+dy
            in_frame = (tx >= 0) & (tx <= 799) & (ty >= 0) & (ty <= 649)
            target_roi = cv2.remap(roi.astype(np.uint8), tx, ty, cv2.INTER_NEAREST).astype(bool)
            valid = roi & in_frame & target_roi
            occluded = cv2.remap(cover, tx, ty, cv2.INTER_NEAREST).astype(bool) & valid
            f, r, _, _ = pair(tensor(a), tensor(b))
            pred = f[0].permute(1, 2, 0).cpu().numpy()
            gt = np.empty_like(pred); gt[..., 0] = dx; gt[..., 1] = dy
            result = region_flow_metrics(pred, gt, valid, occluded)
            records.append({"frame": int(index), "shift": [dx, dy], "metrics": result})
        print(f"synthetic {flow_name(args)} source frame {index}", flush=True)
    fields = ["AEPE_px", "mean_AE_deg", "magnitude_MAE_px", "relative_EPE_eps_0.01",
              "KITTI_rule_outlier_pct", "BadPix_1px_pct", "BadPix_3px_pct", "BadPix_5px_pct"]
    aggregate = {}
    for region in records[0]["metrics"]:
        rows = [r["metrics"][region] for r in records if r["metrics"][region]["count"]]
        count = sum(r["count"] for r in rows)
        aggregate[region] = {"count": count, **{key: sum(r[key]*r["count"] for r in rows)/count
                                                 for key in fields}} if count else {"count": 0}
    save_json(args.run/f"{flow_name(args)}_synthetic.json",
              {"dataset": "48 controlled translations of 8 rectified Orbbec images with known occluders",
               "is_official_benchmark": False, "aggregate": aggregate, "cases": records})


def refine(args):
    from stereo_center.ffs_inference import load_ffs
    from evaluate_ffs_temporal_warmstart import native_ffs
    start = time.perf_counter()
    meta = json.loads((args.run/"manifest.json").read_text())
    rgb, right, raw = [np.load(args.run/f"{name}.npy", mmap_mode="r") for name in ("rgb", "right", "raw_disp")]
    roi = scalar_image(np.load(args.run/"roi.npy")) > .5
    name = args.group
    is_control = name == "control"
    if not is_control:
        fm = json.loads((args.run/f"{name}_flow_summary.json").read_text())
        if fm["pair_hash"] != meta["pair_hash"]:
            raise RuntimeError("Flow PTS hash mismatch")
        flows = np.load(args.run/f"{name}_flow.npy", mmap_mode="r")
    output = np.lib.format.open_memmap(args.run/f"{name}_depth.npy", mode="w+", dtype="float32", shape=raw.shape)
    writer = video_writer(args.run/f"{name}_depth.mp4", meta["fps"])
    ffs = load_ffs(weights_dir=args.root/"CenterDepth/weights/fast_foundation_stereo/23-36-37",
                   ffs_root=args.root/"Fast-FoundationStereo", valid_iters=5)
    elapsed, accepted, weight_means = [], [], []
    try:
        for t in range(meta["n"]):
            d = scalar_image(raw[t]); a, b = tensor(rgb[t]), tensor(right[t])
            if t == 0:
                final = d
            else:
                init = d
                if not is_control:
                    f, r = [torch.from_numpy(np.array(flows[t-1, k]))[None].cuda() for k in (0, 1)]
                    previous = scalar_image(raw[t-1])
                    prior, inside = backward_warp(previous, r)
                    prev_ok = roi & (previous >= meta["fxB"]/20) & (previous <= meta["fxB"]/.3)
                    sampled_ok, _ = backward_warp(prev_ok.float(), r)
                    ok = roi & inside & (sampled_ok >= 1-1e-5) & torch.isfinite(prior)
                    ok &= (d >= meta["fxB"]/20) & (d <= meta["fxB"]/.3)
                    ok &= (prior-d).abs() <= torch.maximum(torch.full_like(d, 3), d*.15)
                    alignment = temporal_alignment_mask(a, tensor(rgb[t-1]), f, r, photo_tol=25.)
                    weight = .75*alignment*ok.float()
                    if meta["left_pts_us"][t]-meta["left_pts_us"][t-1] > 100000:
                        weight.zero_()
                    init = d*(1-weight)+prior*weight
                    accepted.append(float((weight > 0).float().mean()))
                    weight_means.append(float(weight.mean()))
                final, seconds = native_ffs(ffs, a, b, init)
                elapsed.append(seconds)
            output[t] = (meta["fxB"]/final.clamp_min(.001))[0, 0].cpu().numpy()
            draw_depth(writer, output[t])
            if (t+1) % 100 == 0:
                print(f"refine {name} {t+1}/{meta['n']}", flush=True)
    finally:
        writer.release(); output.flush()
    save_json(args.run/f"{name}_refine_summary.json",
              {"frames": meta["n"], "pair_hash": meta["pair_hash"], "weight": 0 if is_control else .75,
               "forward_seconds": elapsed, "nonzero_weight_ratio": float(np.mean(accepted)) if accepted else 0,
               "mean_weight": float(np.mean(weight_means)) if weight_means else 0,
               "wall_seconds": time.perf_counter()-start})


def refine_geometry(args):
    """Use a metric PnP pose and z-buffered 3-D reprojection as FFS init_disp."""
    from stereo_center.ffs_inference import load_ffs
    from evaluate_ffs_temporal_warmstart import native_ffs
    start = time.perf_counter()
    meta = json.loads((args.run/"manifest.json").read_text())
    rgb, right, raw = [np.load(args.run/f"{name}.npy", mmap_mode="r") for name in ("rgb", "right", "raw_disp")]
    flow_path = args.run/"roma_fast_flow.npy"
    if not flow_path.exists():
        raise RuntimeError("RoMa fast forward flow cache is required for geometric PnP")
    flows = np.load(flow_path, mmap_mode="r")
    rect = calib.compute_rectification_maps(
        calib.load_orbbec_calibration(meta["calibration"]), (800, 650)
    )
    camera_matrix = np.asarray(rect["P1"][:, :3], dtype=np.float64)
    roi = np.load(args.run/"roi.npy").astype(bool)
    name = args.group
    output = np.lib.format.open_memmap(args.run/f"{name}_depth.npy", mode="w+", dtype="float32", shape=raw.shape)
    writer = video_writer(args.run/f"{name}_depth.mp4", meta["fps"])
    ffs = load_ffs(weights_dir=args.root/"CenterDepth/weights/fast_foundation_stereo/23-36-37",
                   ffs_root=args.root/"Fast-FoundationStereo", valid_iters=5)
    elapsed, accepts, covers, inlier_ratios, reprojection_rmses = [], [], [], [], []
    translations, rotations, rejected_poses = [], [], 0
    try:
        for t in range(meta["n"]):
            d = scalar_image(raw[t])
            a, b = tensor(rgb[t]), tensor(right[t])
            if t == 0:
                final = d
            else:
                previous_depth = meta["fxB"] / np.maximum(np.array(raw[t-1]), .001)
                current_depth = meta["fxB"] / np.maximum(np.array(raw[t]), .001)
                source_valid = roi & np.isfinite(previous_depth) & (previous_depth >= .3) & (previous_depth <= 20.)
                flow = np.moveaxis(np.array(flows[t-1, 0]), 0, -1)
                pose = estimate_relative_pose_pnp(previous_depth, flow, camera_matrix, source_valid)
                prior_depth = np.zeros_like(previous_depth)
                prior_valid = np.zeros_like(roi)
                if pose is not None:
                    rotation_deg = float(np.rad2deg(np.arccos(np.clip((np.trace(pose.rotation)-1.)/2., -1., 1.))))
                    translation_m = float(np.linalg.norm(pose.translation_m))
                    # Abrupt poses generally indicate a failed RANSAC consensus, not head motion at 30 Hz.
                    if pose.inlier_ratio >= .20 and pose.reprojection_rmse_px <= 2.0 and rotation_deg <= 20. and translation_m <= .5:
                        prior_depth, prior_valid, _rigid_error = reproject_depth_zbuffer(
                            previous_depth, pose, camera_matrix, source_valid, roi,
                            previous_to_current_flow=flow, max_flow_disagreement_px=2.0,
                        )
                        inlier_ratios.append(pose.inlier_ratio)
                        reprojection_rmses.append(pose.reprojection_rmse_px)
                        translations.append(translation_m)
                        rotations.append(rotation_deg)
                    else:
                        rejected_poses += 1
                else:
                    rejected_poses += 1
                prior_valid &= np.isfinite(prior_depth) & (prior_depth >= .3) & (prior_depth <= 20.)
                prior_valid &= np.abs(prior_depth-current_depth) <= np.maximum(.20, .15*current_depth)
                prior_disp = meta["fxB"] / np.maximum(prior_depth, .001)
                weight = float(args.max_prior_weight) * prior_valid.astype(np.float32)
                if meta["left_pts_us"][t]-meta["left_pts_us"][t-1] > 100000:
                    weight.fill(0.)
                init = d*(1-scalar_image(weight))+scalar_image(prior_disp)*scalar_image(weight)
                final, seconds = native_ffs(ffs, a, b, init)
                elapsed.append(seconds)
                accepts.append(float(prior_valid.mean()))
                covers.append(float((prior_depth > 0).mean()))
            output[t] = (meta["fxB"]/final.clamp_min(.001))[0, 0].cpu().numpy()
            draw_depth(writer, output[t])
            if (t+1) % 100 == 0:
                print(f"refine geometry {name} {t+1}/{meta['n']}", flush=True)
    finally:
        writer.release(); output.flush()
    save_json(args.run/f"{name}_refine_summary.json", {
        "frames": meta["n"], "pair_hash": meta["pair_hash"], "max_prior_weight": float(args.max_prior_weight),
        "forward_seconds": elapsed, "accepted_ratio": float(np.mean(accepts)) if accepts else 0,
        "zbuffer_coverage_ratio": float(np.mean(covers)) if covers else 0,
        "pnp_inlier_ratio": samples_summary([np.asarray(inlier_ratios)]),
        "pnp_reprojection_rmse_px": samples_summary([np.asarray(reprojection_rmses)]),
        "translation_norm_m": samples_summary([np.asarray(translations)]),
        "rotation_deg": samples_summary([np.asarray(rotations)]),
        "rejected_or_missing_pose_frames": rejected_poses,
        "wall_seconds": time.perf_counter()-start,
    })


def evaluate(args):
    meta = json.loads((args.run/"manifest.json").read_text())
    raw = np.load(args.run/"raw_disp.npy", mmap_mode="r")
    rgb = np.load(args.run/"rgb.npy", mmap_mode="r")
    names = ["raw", "control"]
    names.extend(name for name in ("sea", "roma_fast", "roma_geometry", "roma_base")
                 if (args.run/f"{name}_depth.npy").exists())
    results = {k: np.load(args.run/f"{k}_depth.npy", mmap_mode="r") for k in names if k != "raw"}
    flows = {name: np.load(args.run/f"{name}_flow.npy", mmap_mode="r")
             for name in names if (args.run/f"{name}_flow.npy").exists()}
    roi = np.load(args.run/"roi.npy"); roi_t = scalar_image(roi) > .5
    changes = {k: [] for k in names}; motion = {f: {k: [] for k in names} for f in flows}
    photos = {f: [] for f in flows}; cycles = {f: [] for f in flows}
    failure = {k: 0 for k in names}; valid_counts = {k: 0 for k in names}
    total_reference = common_count = 0
    flow_common_counts = {flow_name: 0 for flow_name in flows}
    prev = None
    def good(x):
        return np.isfinite(x) & (x >= .3) & (x <= 20)
    for t in range(meta["n"]):
        current = {k: (meta["fxB"]/np.array(raw[t]).clip(.001) if k == "raw" else np.array(results[k][t])) for k in names}
        for k in names:
            valid_counts[k] += int((good(current[k]) & roi).sum())
        if prev is not None:
            reference = roi & good(current["raw"]) & good(prev["raw"])
            common = reference.copy()
            for k in names:
                ok = good(current[k]) & good(prev[k])
                failure[k] += int((reference & ~ok).sum()); common &= ok
            total_reference += int(reference.sum()); common_count += int(common.sum())
            grid = common[::8, ::8]
            for k in names:
                changes[k].append(np.abs(current[k]-prev[k])[::8, ::8][grid])
            for f in flows:
                forward, back = [torch.from_numpy(np.array(flows[f][t-1, j]))[None].cuda() for j in (0, 1)]
                sampled_roi, inside = backward_warp(roi_t.float(), back)
                base = roi_t & inside & (sampled_roi > 1-1e-5)
                wrgb, _ = backward_warp(tensor(rgb[t-1]), back)
                cycle, _ = backward_warp(forward, back)
                sampled = base[0, 0, ::8, ::8]
                photos[f].append((wrgb-tensor(rgb[t])).abs().mean(1)[0, ::8, ::8][sampled].cpu().numpy())
                cycles[f].append(torch.linalg.vector_norm(cycle+back, dim=1)[0, ::8, ::8][sampled].cpu().numpy())
                m = base
                warped = {}
                for k in names:
                    source_ok, _ = backward_warp(scalar_image(good(prev[k])), back)
                    m &= source_ok > 1-1e-5
                    m &= scalar_image(good(current[k])) > .5
                    warped[k], _ = backward_warp(scalar_image(prev[k]), back)
                mask = m[0, 0, ::8, ::8]
                flow_common_counts[f] += int(mask.sum())
                for k in names:
                    residual = (warped[k]-scalar_image(current[k])).abs()
                    motion[f][k].append(residual[0, 0, ::8, ::8][mask].cpu().numpy())
        prev = current
        if (t+1) % 200 == 0:
            print(f"evaluate {t+1}/{meta['n']}", flush=True)
    save_json(args.run/"comparison.json", {
        "frames": meta["n"], "spatial_stride": 8, "units": "depth=m; flow=px; photo=RGB 0-255",
        "reference_pair_pixel_count": total_reference, "common_pair_pixel_count": common_count,
        "flow_common_pair_pixel_count": flow_common_counts,
        "failure_on_reference": failure, "valid_pixel_counts": valid_counts,
        "same_pixel_change": {k: samples_summary(v) for k, v in changes.items()},
        "flow_compensated_depth_change": {f: {k: samples_summary(v) for k, v in groups.items()} for f, groups in motion.items()},
        "photo_proxy": {k: samples_summary(v) for k, v in photos.items()},
        "fb_cycle_proxy": {k: samples_summary(v) for k, v in cycles.items()},
        "note": "Real video has no flow/depth GT; photo/cycle and temporal residuals are proxies, not EPE or accuracy."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "flow", "synthetic", "refine", "refine_geometry", "evaluate"])
    parser.add_argument("--root", type=Path, default=Path("/home/opsuser/BothEyesDepth"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--backend", choices=["sea", "roma"], default="roma")
    parser.add_argument("--setting", choices=["fast", "base"], default="fast")
    parser.add_argument("--group", default="roma_fast")
    parser.add_argument("--max-prior-weight", type=float, default=.75)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    torch.set_num_threads(8); cv2.setNumThreads(4)
    {"prepare": prepare, "flow": estimate_flow, "synthetic": synthetic,
     "refine": refine, "refine_geometry": refine_geometry,
     "evaluate": evaluate}[arguments.stage](arguments)
