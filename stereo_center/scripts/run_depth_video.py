#!/usr/bin/env python
"""批量生成中心视角深度视频：校正 → 批量 WAFT 双向视差 → 中心融合 → 深度帧 → mp4。

用法示例（在 stereo_center/ 目录下）：
    conda run -n waft python scripts/run_depth_video.py \
        --video ../dataset/xxx/output.mp4 --calib ../dataset/xxx/calibration.json \
        --scale 0.5 --batch-size 4 --outdir outputs/depth_video

模型只加载一次，视频帧按 batch 送入 WAFT 前向（s0.5 为 direct 路径，
实测 B=4 时 960x600 前向约 0.93s/批，显存峰值 ~9GB）。
"""

from __future__ import annotations

import argparse
import json
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

from stereo_center import calib, softsplat, stereo_backend  # noqa: E402
from stereo_center.pipeline import photometric_align_right  # noqa: E402
from stereo_center.visualize import colorize_depth  # noqa: E402


def resolve_weights_dir(explicit: str | None, backend: str) -> Path:
    if explicit:
        return Path(explicit)
    env = "WAFT_WEIGHTS_DIR" if backend == "waft" else "S2M2_WEIGHTS_DIR"
    if env in __import__("os").environ:
        return Path(__import__("os").environ[env])
    repo_root = PROJECT_ROOT.parent
    sub = "waft" if backend == "waft" else "pretrain_weights"
    for c in (repo_root / "weights" / sub, PROJECT_ROOT / "weights" / sub):
        if c.exists():
            return c
    raise FileNotFoundError(f"未找到权重目录（{backend}），请用 --weights 或环境变量 {env}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量中心深度视频生成")
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
    parser.add_argument("--save-frames-every", type=int, default=50, help="每隔 N 帧存一张深度 PNG（0=不存）")
    parser.add_argument("--video-name", type=str, default="depth_video.mp4")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backend = args.stereo_backend

    # 1) 标定 + 校正表（所有帧共用）
    cal = calib.load_vdego_calibration(args.calib)
    out_size = (
        max(32, int(cal["resolution"][0] * args.scale)),
        max(32, int(cal["resolution"][1] * args.scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    fx, fy = rect["P1"][0, 0], rect["P1"][1, 1]
    cx, cy = rect["P1"][0, 2], rect["P1"][1, 2]
    baseline = rect["baseline"]
    H, W = out_size[1], out_size[0]
    print(f"[rect] {W}x{H}, fx={fx:.1f}, baseline={baseline:.4f} m")

    # 2) 模型（只加载一次）
    weights_dir = resolve_weights_dir(args.weights, backend)
    print(f"[weights] {weights_dir}")
    model = stereo_backend.load(backend, args.model_type, str(weights_dir), args.device, num_refine=3)

    # 3) 视频读取 + 输出
    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else float(src_fps)
    end = args.end_frame if args.end_frame >= 0 else n_total
    if args.max_frames > 0:
        end = min(end, args.start_frame + args.max_frames)
    writer = cv2.VideoWriter(
        str(outdir / args.video_name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    print(f"[video] 共 {n_total} 帧，处理 [{args.start_frame}, {end})，输出 fps={fps:.2f}")

    t_all = time.time()
    frame_idx = args.start_frame
    processed = 0
    while frame_idx < end:
        # 读一批帧
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

        # 校正（每帧 remap）并组 batch
        left_t = torch.zeros(B, 3, H, W)
        right_t = torch.zeros(B, 3, H, W)
        bgr_pairs = []
        for b, img in enumerate(batch_frames):
            l_bgr, r_bgr = img[:, : img.shape[1] // 2], img[:, img.shape[1] // 2 :]
            rL, rR = calib.rectify_pair(l_bgr, r_bgr, rect)
            bgr_pairs.append((rL, rR))
            left_t[b] = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            right_t[b] = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

        # 批量双向视差
        from stereo_center import waft_inference

        t0 = time.time()
        dL, dR, occL, occR, confL, confR, _ = waft_inference.run_stereo_matching_bi_batch(
            model, left_t, right_t, args.device,
            hiera=args.hiera, conf_mode=args.conf, occ_mode=args.occ,
        )
        infer_s = time.time() - t0

        # 逐帧融合 + 深度帧
        for b in range(B):
            rL_bgr, rR_bgr = bgr_pairs[b]
            rR_f = photometric_align_right(rL_bgr, rR_bgr)
            left_f = torch.from_numpy(cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
            right_f = torch.from_numpy(cv2.cvtColor(rR_f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0)
            dl = dL[b].unsqueeze(0).unsqueeze(0)
            dr = dR[b].unsqueeze(0).unsqueeze(0)
            cl = confL[b].unsqueeze(0).unsqueeze(0)
            cr = confR[b].unsqueeze(0).unsqueeze(0)
            ol = occL[b].unsqueeze(0).unsqueeze(0)
            orr = occR[b].unsqueeze(0).unsqueeze(0)
            rgb, dep, valid = softsplat.center_view(
                left_f, right_f, dl, cl, ol, fx=fx, baseline=baseline,
                disp_right=dr, conf_right=cr, occ_right=orr,
                edge_k=1.5, blend="softz", weight_mode="expdecay", weight_k=4.0,
                depth_z=bool(args.depth_z), depth_z_thresh=0.05, depth_z_power=2.0,
                color_tol=args.color_tol,
            )
            rgb_np = rgb[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
            dep_np = dep[0, 0].numpy()
            valid_np = valid[0, 0].numpy().astype(bool)
            rgb_np, dep_np, valid_np = softsplat.fill_disocclusion(rgb_np, dep_np, valid_np)
            depth_img = colorize_depth(dep_np, valid_np)  # BGR jet
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
                f"推理 {infer_s:.1f}s/批，吞吐 {rate:.2f} 帧/s，ETA {eta/60:.1f} min",
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
    }
    (outdir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {processed} 帧，总耗时 {total_s:.1f}s（{total_s/max(processed,1):.2f}s/帧）")
    print(f"[out] {outdir / args.video_name}")


if __name__ == "__main__":
    main()
