#!/usr/bin/env python
"""StereoAnyVideo 中心深度视频生成：校正 → SAV 视频级双向视差 → 中心融合 → 深度帧 → mp4。

与 run_depth_video.py 的区别：立体匹配改用 StereoAnyVideo（ICCV 2025，
时间一致视频级匹配），不再逐帧推理；其余完全复用现有管线
（calib / photometric_align_right / softsplat.center_view /
fill_disocclusion_torch / colorize_depth_log）。

用法示例（在 stereo_center/ 目录下，服务器 sav conda 环境）：
    conda run -n sav python scripts/run_depth_video_sav.py \
        --video ../dataset/xxx/output.mp4 --calib ../dataset/xxx/calibration.json \
        --scale 0.5 --outdir outputs/depth_video_sav/xxx

默认 SF（SceneFlow）checkpoint；StereoAnyVideo 官方只输出左视差，
因此默认双向推理（左→右 + 右→左）获得右参考视差，与现有融合逻辑对齐。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, sav_inference, softsplat  # noqa: E402
from stereo_center.pipeline import photometric_align_right  # noqa: E402
from stereo_center.visualize import (  # noqa: E402
    colorize_depth_log,
    make_depth_colorbar_log,
)
from stereo_center.orbbec import (  # noqa: E402
    load_pts_us,
    match_left_to_right_pts,
    pts_metadata_mismatch,
    pts_sidecar_path,
)


def left_view_depth_from_disparity(
    disparity: torch.Tensor,
    fx: float,
    baseline: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert SAV left-reference disparity directly to metric left-camera depth."""
    disp = disparity.to(device).float()
    valid = torch.isfinite(disp) & (disp >= 0.5)
    depth = float(fx) * float(baseline) / disp.clamp_min(1e-6)
    return depth.unsqueeze(1), valid.unsqueeze(1)


def resolve_sav_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if "SAV_ROOT" in os.environ:
        return Path(os.environ["SAV_ROOT"]).resolve()
    for c in (
        PROJECT_ROOT.parent / "stereoanyvideo",
        Path.home() / "BothEyesDepth" / "stereoanyvideo",
    ):
        if (c / "models").exists():
            return c
    raise FileNotFoundError(
        "未找到 stereoanyvideo 仓库，请用 --sav-root 或环境变量 SAV_ROOT 指定"
    )


def calibration_loader_for_path(path: str | Path):
    return (
        calib.load_orbbec_calibration
        if Path(path).suffix.lower() in {".yaml", ".yml"}
        else calib.load_vdego_calibration
    )


def read_rectified_frames(
    cap: cv2.VideoCapture, rect: dict, start: int, count: int
) -> list:
    """从 start 开始连续读取 count 帧，返回校正后 (left_bgr, right_bgr) 列表。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(count):
        ok, img = cap.read()
        if not ok:
            break
        l_bgr = img[:, : img.shape[1] // 2]
        r_bgr = img[:, img.shape[1] // 2 :]
        out.append(calib.rectify_pair(l_bgr, r_bgr, rect))
    return out


def read_rectified_split_frames(
    cap_left: cv2.VideoCapture,
    cap_right: cv2.VideoCapture,
    rect: dict,
    paired_indices: list[tuple[int, int]],
    start: int,
    count: int,
) -> list:
    """Read split Orbbec streams by matched PTS frame pairs."""
    out = []
    for left_idx, right_idx in paired_indices[start : start + count]:
        cap_left.set(cv2.CAP_PROP_POS_FRAMES, left_idx)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, right_idx)
        ok_left, left_bgr = cap_left.read()
        ok_right, right_bgr = cap_right.read()
        if not ok_left or not ok_right:
            break
        out.append(calib.rectify_pair(left_bgr, right_bgr, rect))
    return out


def pairs_to_tensors(
    bgr_pairs: list, size: tuple
) -> tuple[torch.Tensor, torch.Tensor]:
    """校正后 BGR 对列表 → (N,3,H,W) float32 0-255 RGB CPU 张量。"""
    H, W = size
    N = len(bgr_pairs)
    left = torch.zeros(N, 3, H, W)
    right = torch.zeros(N, 3, H, W)
    for i, (rL, rR) in enumerate(bgr_pairs):
        left[i] = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        right[i] = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    return left, right


def fuse_batch(
    bgr_pairs: list,
    dl: np.ndarray,
    dr: np.ndarray | None,
    rect: dict,
    fx: float,
    baseline: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    """保留帧的视差 → center_view 融合 + 遮挡填充（GPU 批量）。

    Returns:
        dep_b: (K, 1, H, W) float32 GPU 中心深度（米，已填充）；
        valid_b: (K, 1, H, W) bool GPU 有效掩码。
    """
    dev = args.device
    K = len(bgr_pairs)
    rgb_list, dep_list, valid_list = [], [], []
    for b in range(K):
        rL_bgr, rR_bgr = bgr_pairs[b]
        rR_f = photometric_align_right(rL_bgr, rR_bgr)
        left_f = (
            torch.from_numpy(cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1).float().unsqueeze(0).to(dev)
        )
        right_f = (
            torch.from_numpy(cv2.cvtColor(rR_f, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1).float().unsqueeze(0).to(dev)
        )
        d_l = torch.from_numpy(dl[b]).float().unsqueeze(0).unsqueeze(0).to(dev)
        _, _, Wd = dl.shape
        x = torch.arange(Wd, device=dev).view(1, 1, 1, Wd)
        occ_l = ((d_l >= 0.5) & (x - d_l >= 0) & (d_l < Wd - 1)).float()
        conf_l = torch.ones_like(occ_l)
        disp_r = conf_r = occ_r = None
        if dr is not None:
            d_r = torch.from_numpy(dr[b]).float().unsqueeze(0).unsqueeze(0).to(dev)
            occ_r = ((d_r >= 0.5) & (x - d_r >= 0) & (d_r < Wd - 1)).float()
            conf_r = torch.ones_like(occ_r)
            disp_r = d_r
        rgb, dep, valid = softsplat.center_view(
            left_f, right_f, d_l, conf_l, occ_l, fx=fx, baseline=baseline,
            disp_right=disp_r, conf_right=conf_r, occ_right=occ_r,
            edge_k=1.5, blend="softz", weight_mode="expdecay", weight_k=4.0,
            depth_z=bool(args.depth_z), depth_z_thresh=0.05, depth_z_power=2.0,
            color_tol=args.color_tol,
        )
        rgb_list.append(rgb)
        dep_list.append(dep)
        valid_list.append(valid)
    rgb_b = torch.cat(rgb_list, dim=0)
    dep_b = torch.cat(dep_list, dim=0)
    valid_b = torch.cat(valid_list, dim=0)
    _rgb_b, dep_b, valid_b = softsplat.fill_disocclusion_torch(rgb_b, dep_b, valid_b)
    return dep_b, valid_b


def main() -> None:
    parser = argparse.ArgumentParser(description="StereoAnyVideo 中心深度视频生成")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video-right", type=str, default=None)
    parser.add_argument("--left-pts", type=str, default=None, help="左视频 timestamp_us CSV（默认自动查找）")
    parser.add_argument("--right-pts", type=str, default=None)
    parser.add_argument("--pts-match-tolerance-ms", type=float, default=8.0)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--scale", type=float, default=0.5, help="校正输出缩放（建议 0.5）")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="-1=到视频末尾")
    parser.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理 N 帧（调试用）")
    parser.add_argument("--fps", type=float, default=0.0, help="输出视频 fps（默认取源视频）")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/depth_video_sav"))
    parser.add_argument("--sav-root", type=str, default=None, help="StereoAnyVideo 仓库根目录")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="StereoAnyVideo checkpoint（默认 sav_root/checkpoints/StereoAnyVideo_SF.pth）",
    )
    parser.add_argument("--iters", type=int, default=20, help="迭代次数（官方 demo 默认 20）")
    parser.add_argument("--seg-len", type=int, default=400, help="单段视频帧数（时间窗口=20）")
    parser.add_argument("--overlap", type=int, default=20, help="段间重叠帧数（前后各一半）")
    parser.add_argument("--bi", type=int, default=1, choices=[0, 1], help="双向推理（交换输入得到右视差）")
    parser.add_argument(
        "--output-view", type=str, default="center", choices=["center", "left"],
        help="输出视角：center=中心合成；left=左相机视角，不做中心融合",
    )
    parser.add_argument("--amp", type=int, default=0, choices=[0, 1], help="混合精度推理（实验性，可能数值不稳）")
    parser.add_argument("--tf32", type=int, default=0, choices=[0, 1], help="A100 TF32 加速（实验性：迭代匹配数值不稳，默认关）")
    parser.add_argument("--bench", type=int, default=0, choices=[0, 1], help="cudnn benchmark 自动选卷积算法（实验性）")
    parser.add_argument("--fusion-batch", type=int, default=4, help="融合阶段的 GPU batch")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--color-tol", type=float, default=15.0, help="softz 颜色冲突阈值")
    parser.add_argument("--depth-z", type=int, default=1, choices=[0, 1], help="深度 hard z-buffer")
    parser.add_argument("--dmin-m", type=float, default=0.3, help="对数米制色阶下限（米）")
    parser.add_argument("--dmax-m", type=float, default=20.0, help="对数米制色阶上限（米）")
    parser.add_argument("--save-frames-every", type=int, default=50, help="每隔 N 帧存一张深度 PNG（0=不存）")
    parser.add_argument("--video-name", type=str, default="depth_video.mp4")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cal = calibration_loader_for_path(args.calib)(args.calib)
    out_size = (
        max(32, int(cal["resolution"][0] * args.scale)),
        max(32, int(cal["resolution"][1] * args.scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    fx = rect["P1"][0, 0]
    baseline = rect["baseline"]
    H, W = out_size[1], out_size[0]
    print(f"[rect] {W}x{H}, fx={fx:.1f}, baseline={baseline:.4f} m")

    sav_root = resolve_sav_root(args.sav_root)
    ckpt = Path(args.ckpt).resolve() if args.ckpt else sav_root / "checkpoints" / "StereoAnyVideo_SF.pth"
    print(f"[sav] root={sav_root}\n[sav] ckpt={ckpt}")
    sav = sav_inference.load_sav(
        sav_root, ckpt, args.device, amp=bool(args.amp),
        tf32=bool(args.tf32), bench=bool(args.bench),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开左/SBS 视频: {args.video}")
    cap_right = cv2.VideoCapture(args.video_right) if args.video_right else None
    if cap_right is not None and not cap_right.isOpened():
        raise RuntimeError(f"无法打开右视频: {args.video_right}")
    n_left_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paired_indices = None
    if cap_right is not None:
        left_pts_path = Path(args.left_pts) if args.left_pts else pts_sidecar_path(args.video)
        right_pts_path = Path(args.right_pts) if args.right_pts else pts_sidecar_path(args.video_right)
        if not left_pts_path.is_file() or not right_pts_path.is_file():
            raise FileNotFoundError(
                "独立左右视频必须提供 PTS CSV；可用 --left-pts/--right-pts 显式指定"
            )
        left_pts = load_pts_us(left_pts_path)
        right_pts = load_pts_us(right_pts_path)
        n_right_total = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        if pts_metadata_mismatch(len(left_pts), len(right_pts), n_left_total, n_right_total):
            print(
                f"[sync] 容器帧数 {n_left_total}/{n_right_total} 与 PTS "
                f"{len(left_pts)}/{len(right_pts)} 不一致，按 PTS 配对"
            )
        paired_indices, sync_offsets_us = match_left_to_right_pts(
            left_pts, right_pts, round(args.pts_match_tolerance_ms * 1000),
        )
        if not paired_indices:
            raise ValueError("没有 PTS 差值在容差内的双目帧对")
        n_total = len(paired_indices)
        print(
            f"[sync] PTS 配对 {n_total}/{n_left_total} 左帧，"
            f"中位偏差={np.median(sync_offsets_us) / 1000:.3f}ms，"
            f"最大偏差={sync_offsets_us.max() / 1000:.3f}ms"
        )
    else:
        n_total = n_left_total
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else float(src_fps)
    end = args.end_frame if args.end_frame >= 0 else n_total
    if args.max_frames > 0:
        end = min(end, args.start_frame + args.max_frames)
    seg_len = max(30, int(args.seg_len))
    overlap = max(0, min(int(args.overlap), seg_len // 2))
    print(f"[video] 共 {n_total} 帧，处理 [{args.start_frame}, {end})，"
          f"seg_len={seg_len} overlap={overlap} bi={bool(args.bi)}，输出 fps={fps:.2f}")

    cv2.imwrite(
        str(outdir / "colorbar.png"),
        make_depth_colorbar_log(args.dmin_m, args.dmax_m, height=H),
    )
    writer = cv2.VideoWriter(
        str(outdir / args.video_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    t_all = time.time()
    t_stereo = 0.0
    t_fusion = 0.0
    frame_idx = args.start_frame
    processed = 0
    stride = max(1, seg_len - 2 * overlap)
    while frame_idx < end:
        seg_start = frame_idx
        seg_end = min(frame_idx + seg_len, end)
        if paired_indices is None:
            bgr_pairs = read_rectified_frames(cap, rect, seg_start, seg_end - seg_start)
        else:
            bgr_pairs = read_rectified_split_frames(
                cap, cap_right, rect, paired_indices, seg_start, seg_end - seg_start
            )
        if not bgr_pairs:
            break
        orig_len = len(bgr_pairs)
        left_t, right_t = pairs_to_tensors(bgr_pairs, (H, W))

        t0 = time.time()
        dL, dR = sav.run_segment(
            left_t, right_t, iters=args.iters, target_len=seg_len,
            bidirectional=bool(args.bi), verbose=True,
        )
        t_stereo += time.time() - t0

        k0 = 0 if seg_start == args.start_frame else overlap
        k1 = orig_len if seg_end == end else orig_len - overlap
        if k1 <= k0:
            frame_idx = seg_end
            continue
        kept_pairs = bgr_pairs[k0:k1]
        dL_k = dL[k0:k1]
        dR_k = dR[k0:k1] if dR is not None else None

        for b in range(0, len(kept_pairs), args.fusion_batch):
            chunk = kept_pairs[b : b + args.fusion_batch]
            t0 = time.time()
            if args.output_view == "left":
                dep_b, valid_b = left_view_depth_from_disparity(
                    torch.from_numpy(dL_k[b : b + args.fusion_batch]),
                    fx,
                    baseline,
                    args.device,
                )
            else:
                dep_b, valid_b = fuse_batch(
                    chunk,
                    dL_k[b : b + args.fusion_batch],
                    dR_k[b : b + args.fusion_batch] if dR_k is not None else None,
                    rect,
                    fx,
                    baseline,
                    args,
                )
            t_fusion += time.time() - t0
            for i in range(dep_b.shape[0]):
                dep_np = dep_b[i, 0].cpu().numpy()
                valid_np = valid_b[i, 0].cpu().numpy().astype(bool)
                depth_img = colorize_depth_log(dep_np, valid_np, args.dmin_m, args.dmax_m)
                writer.write(depth_img)
                if args.save_frames_every > 0 and processed % args.save_frames_every == 0:
                    cv2.imwrite(str(outdir / f"frame_{processed:05d}.png"), depth_img)
                processed += 1

        if seg_end == end:
            frame_idx = end  # 已覆盖到末尾，结束循环
        else:
            frame_idx = seg_start + stride
        elapsed = time.time() - t_all
        rate = processed / max(elapsed, 1e-6)
        eta = (end - args.start_frame - processed) / rate if rate > 0 else 0
        print(
            f"[progress] {processed}/{end - args.start_frame} 帧，"
            f"吞吐 {rate:.2f} 帧/s，ETA {eta/60:.1f} min",
            flush=True,
        )

    writer.release()
    cap.release()
    if cap_right is not None:
        cap_right.release()
    total_s = time.time() - t_all
    stats = {
        "video": str(args.video),
        "scale": args.scale,
        "stereo_backend": "stereoanyvideo",
        "model": "StereoAnyVideo_SF",
        "ckpt": str(ckpt),
        "iters": args.iters,
        "seg_len": seg_len,
        "overlap": overlap,
        "bidirectional": bool(args.bi),
        "output_view": args.output_view,
        "amp": bool(args.amp),
        "tf32": bool(args.tf32),
        "bench": bool(args.bench),
        "fusion_batch": args.fusion_batch,
        "start_frame": args.start_frame,
        "end_frame": frame_idx,
        "n_frames": processed,
        "fps": fps,
        "size": [W, H],
        "total_seconds": round(total_s, 2),
        "avg_seconds_per_frame": round(total_s / max(processed, 1), 4),
        "color_tol": args.color_tol,
        "depth_z": bool(args.depth_z),
        "colormap": "log_metric",
        "dmin_m": args.dmin_m,
        "dmax_m": args.dmax_m,
        "stage_stereo_seconds": round(t_stereo, 2),
        "stage_fusion_seconds": round(t_fusion, 2),
        "stage_other_seconds": round(max(total_s - t_stereo - t_fusion, 0), 2),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    timing = {
        "video": str(args.video),
        "video_right": str(args.video_right) if args.video_right else None,
        "backend": "stereoanyvideo",
        "model": "StereoAnyVideo_SF",
        "bidirectional": bool(args.bi),
        "output_view": args.output_view,
        "scale": args.scale,
        "iters": args.iters,
        "seg_len": seg_len,
        "overlap": overlap,
        "n_frames": processed,
        "stage_seconds": {
            "stereo_forward_seconds": round(t_stereo, 6),
            "left_depth_or_fusion_seconds": round(t_fusion, 6),
            "other_seconds": round(max(total_s - t_stereo - t_fusion, 0), 6),
            "total_seconds": round(total_s, 6),
        },
        "avg_seconds_per_frame": round(total_s / max(processed, 1), 6),
    }
    (outdir / "sav_timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {processed} 帧，总耗时 {total_s:.1f}s（{total_s/max(processed,1):.2f}s/帧）")
    print(f"[out] {outdir / 'sav_timing.json'}")
    print(f"[out] {outdir / args.video_name}")


if __name__ == "__main__":
    main()
