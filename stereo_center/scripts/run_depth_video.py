#!/usr/bin/env python
"""批量生成中心视角深度视频：校正 → 批量 WAFT 双向视差 → 中心融合 → 深度帧 → mp4。

用法示例（在 stereo_center/ 目录下）：
    conda run -n waft python scripts/run_depth_video.py \
        --video ../dataset/xxx/output.mp4 --calib ../dataset/xxx/calibration.json \
        --scale 0.5 --batch-size 4 --outdir outputs/depth_video

- 深度值恒为米制：depth = fx * baseline / disparity。
- 色阶为固定对数米制映射（0.3~20m，可调）：场景深度范围大时线性色阶
  顾此失彼，对数映射让近场/背景都有可分辨色带，且跨帧同深度同色。
- 时间维中值滤波（默认 3 帧）抑制 WAFT 逐帧推理的深度抖动/闪烁。
- 遮挡填充与融合都在 GPU 上批量执行。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, softsplat, stereo_backend, waft_inference  # noqa: E402
from stereo_center.pipeline import photometric_align_right  # noqa: E402
from stereo_center.visualize import (  # noqa: E402
    colorize_depth_log,
    make_depth_colorbar_log,
)


def resolve_weights_dir(explicit: str | None, backend: str) -> Path:
    if explicit:
        return Path(explicit)
    env = "WAFT_WEIGHTS_DIR" if backend == "waft" else "S2M2_WEIGHTS_DIR"
    if env in os.environ:
        return Path(os.environ[env])
    repo_root = PROJECT_ROOT.parent
    sub = "waft" if backend == "waft" else "pretrain_weights"
    for c in (repo_root / "weights" / sub, PROJECT_ROOT / "weights" / sub):
        if c.exists():
            return c
    raise FileNotFoundError(f"未找到权重目录（{backend}），请用 --weights 或环境变量 {env}")


def process_batch(
    model,
    bgr_pairs: list,
    rect: dict,
    fx: float,
    baseline: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    """校正后 BGR 对列表 → GPU 批量推理 + 融合 + 遮挡填充。

    Returns:
        dep_b: (B, 1, H, W) float32 GPU 中心深度（米，已填充）；
        valid_b: (B, 1, H, W) bool GPU 有效掩码。
    """
    B = len(bgr_pairs)
    H, W = bgr_pairs[0][0].shape[:2]
    left_t = torch.zeros(B, 3, H, W)
    right_t = torch.zeros(B, 3, H, W)
    for b, (rL, rR) in enumerate(bgr_pairs):
        left_t[b] = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_t[b] = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

    dL, dR, occL, occR, confL, confR, _ = waft_inference.run_stereo_matching_bi_batch(
        model, left_t, right_t, args.device,
        hiera=args.hiera, conf_mode=args.conf, occ_mode=args.occ,
    )
    dev = args.device
    fusion_out = []
    for b in range(B):
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
        dl = dL[b].unsqueeze(0).unsqueeze(0).to(dev)
        dr = dR[b].unsqueeze(0).unsqueeze(0).to(dev)
        cl = confL[b].unsqueeze(0).unsqueeze(0).to(dev)
        cr = confR[b].unsqueeze(0).unsqueeze(0).to(dev)
        ol = occL[b].unsqueeze(0).unsqueeze(0).to(dev)
        orr = occR[b].unsqueeze(0).unsqueeze(0).to(dev)
        rgb, dep, valid = softsplat.center_view(
            left_f, right_f, dl, cl, ol, fx=fx, baseline=baseline,
            disp_right=dr, conf_right=cr, occ_right=orr,
            edge_k=1.5, blend="softz", weight_mode="expdecay", weight_k=4.0,
            depth_z=bool(args.depth_z), depth_z_thresh=0.05, depth_z_power=2.0,
            color_tol=args.color_tol,
        )
        fusion_out.append((rgb, dep, valid))
    rgb_b = torch.cat([x[0] for x in fusion_out], dim=0)
    dep_b = torch.cat([x[1] for x in fusion_out], dim=0)
    valid_b = torch.cat([x[2] for x in fusion_out], dim=0)
    rgb_b, dep_b, valid_b = softsplat.fill_disocclusion_torch(rgb_b, dep_b, valid_b)
    return dep_b, valid_b


def main() -> None:
    parser = argparse.ArgumentParser(description="批量中心深度视频生成（米制色阶）")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--scale", type=float, default=0.5, help="校正输出缩放（建议 0.5）")
    parser.add_argument("--batch-size", type=int, default=4, help="WAFT 前向 batch（s0.5 建议 4）")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1, help="-1=到视频末尾")
    parser.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理 N 帧（调试用）")
    parser.add_argument("--fps", type=float, default=0.0, help="输出视频 fps（默认取源视频）")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/depth_video"))
    parser.add_argument("--stereo-backend", type=str, default="waft", choices=["waft", "s2m2"])
    parser.add_argument("--model-type", type=str, default="DAv2L-5")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hiera", type=str, default="auto", choices=["auto", "direct", "hiera"])
    parser.add_argument("--conf", type=str, default="lr", choices=["info", "ones", "lr"])
    parser.add_argument("--occ", type=str, default="lr", choices=["lr", "visibility"])
    parser.add_argument("--color-tol", type=float, default=15.0, help="softz 颜色冲突阈值")
    parser.add_argument("--depth-z", type=int, default=1, choices=[0, 1], help="深度 hard z-buffer")
    parser.add_argument(
        "--dmin-m", type=float, default=0.3,
        help="对数米制色阶下限（米，默认 0.3）",
    )
    parser.add_argument(
        "--dmax-m", type=float, default=20.0,
        help="对数米制色阶上限（米，默认 20；超出部分饱和为红色）",
    )
    parser.add_argument(
        "--temporal-median", type=int, default=3,
        help="时间维中值滤波窗口（奇数，1=关闭；默认 3，只去孤立尖峰不拖影）",
    )
    parser.add_argument(
        "--temporal-ema", type=float, default=0.0,
        help="时间维 EMA：0=光流运动补偿 EMA（默认，相机运动被补偿、无拖影）；"
        ">0=固定系数（无光流）；1=关闭",
    )
    parser.add_argument("--save-frames-every", type=int, default=50, help="每隔 N 帧存一张深度 PNG（0=不存）")
    parser.add_argument("--video-name", type=str, default="depth_video.mp4")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backend = args.stereo_backend

    cal = calib.load_vdego_calibration(args.calib)
    out_size = (
        max(32, int(cal["resolution"][0] * args.scale)),
        max(32, int(cal["resolution"][1] * args.scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    fx = rect["P1"][0, 0]
    baseline = rect["baseline"]
    H, W = out_size[1], out_size[0]
    print(f"[rect] {W}x{H}, fx={fx:.1f}, baseline={baseline:.4f} m")

    weights_dir = resolve_weights_dir(args.weights, backend)
    print(f"[weights] {weights_dir}")
    model = stereo_backend.load(backend, args.model_type, str(weights_dir), args.device, num_refine=3)

    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else float(src_fps)
    end = args.end_frame if args.end_frame >= 0 else n_total
    if args.max_frames > 0:
        end = min(end, args.start_frame + args.max_frames)
    print(f"[video] 共 {n_total} 帧，处理 [{args.start_frame}, {end})，输出 fps={fps:.2f}")

    cv2.imwrite(
        str(outdir / "colorbar.png"),
        make_depth_colorbar_log(args.dmin_m, args.dmax_m, height=H),
    )
    print(f"[color] 对数米制色阶 {args.dmin_m}~{args.dmax_m}m（全片固定，colorbar.png 已保存）")

    writer = cv2.VideoWriter(
        str(outdir / args.video_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    t_all = time.time()
    t_stereo_fusion_fill = 0.0
    depth_tbuf = deque(maxlen=max(1, args.temporal_median))
    win = max(1, args.temporal_median)
    prev_gray = None
    prev_depth = None
    flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                       poly_n=5, poly_sigma=1.2, flags=0)
    frame_idx = args.start_frame
    processed = 0
    while frame_idx < end:
        batch_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        for _ in range(min(args.batch_size, end - frame_idx)):
            ok, img = cap.read()
            if not ok:
                break
            batch_frames.append(img)
        if not batch_frames:
            break
        B = len(batch_frames)

        bgr_pairs = []
        for img in batch_frames:
            l_bgr, r_bgr = img[:, : img.shape[1] // 2], img[:, img.shape[1] // 2 :]
            bgr_pairs.append(calib.rectify_pair(l_bgr, r_bgr, rect))
        t0 = time.time()
        dep_b, valid_b = process_batch(model, bgr_pairs, rect, fx, baseline, args)
        t_stereo_fusion_fill += time.time() - t0

        for b in range(B):
            rL_bgr, rR_bgr = bgr_pairs[b]
            d_cur = dep_b[b].unsqueeze(0)  # (1,1,H,W) GPU
            v_cur = valid_b[b].unsqueeze(0)
            if win > 1:
                depth_tbuf.append(d_cur)
                if len(depth_tbuf) == win:
                    d_cur = torch.median(torch.stack(list(depth_tbuf)), dim=0).values
            dep_np = d_cur[0, 0].cpu().numpy()
            valid_np = v_cur[0, 0].cpu().numpy().astype(bool)
            # 时间 EMA：默认用光流把上一帧深度 warp 到当前帧再融合（运动补偿）
            if args.temporal_ema != 1.0:
                cur_gray = cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev_depth is not None and prev_gray is not None:
                    if args.temporal_ema > 0.0:
                        alpha = args.temporal_ema
                        warped = prev_depth
                    else:
                        alpha = 0.35
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, cur_gray, None, **flow_params
                        )
                        yy, xx = np.mgrid[0 : dep_np.shape[0], 0 : dep_np.shape[1]]
                        mapx = (xx - flow[..., 0]).astype(np.float32)
                        mapy = (yy - flow[..., 1]).astype(np.float32)
                        warped = cv2.remap(
                            prev_depth, mapx, mapy,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                        )
                    smoothed = alpha * dep_np + (1.0 - alpha) * warped
                    dep_np = np.where(valid_np, smoothed, dep_np)
                prev_gray = cur_gray
                prev_depth = dep_np
            depth_img = colorize_depth_log(dep_np, valid_np, args.dmin_m, args.dmax_m)
            writer.write(depth_img)
            if args.save_frames_every > 0 and processed % args.save_frames_every == 0:
                cv2.imwrite(str(outdir / f"frame_{frame_idx + b:05d}.png"), depth_img)
            processed += 1

        frame_idx += B
        if processed % (args.batch_size * 10) == 0 or frame_idx >= end:
            rate = processed / max(time.time() - t_all, 1e-6)
            eta = (end - frame_idx) / rate if rate > 0 else 0
            print(
                f"[progress] {processed}/{end - args.start_frame} 帧，"
                f"吞吐 {rate:.2f} 帧/s，ETA {eta/60:.1f} min",
                flush=True,
            )

    writer.release()
    cap.release()
    total_s = time.time() - t_all
    stats = {
        "video": str(args.video),
        "scale": args.scale,
        "batch_size": args.batch_size,
        "stereo_backend": backend,
        "model_type": args.model_type,
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
        "temporal_median": win,
        "temporal_ema": args.temporal_ema,
        "stage_stereo_fusion_fill_seconds": round(t_stereo_fusion_fill, 2),
        "stage_other_seconds": round(max(total_s - t_stereo_fusion_fill, 0), 2),
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {processed} 帧，总耗时 {total_s:.1f}s（{total_s/max(processed,1):.2f}s/帧）")
    print(f"[out] {outdir / args.video_name}")


if __name__ == "__main__":
    main()
