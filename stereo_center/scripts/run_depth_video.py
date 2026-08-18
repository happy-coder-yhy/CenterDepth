#!/usr/bin/env python
"""批量生成中心视角深度视频：校正 → 批量 WAFT 双向视差 → 中心融合 → 深度帧 → mp4。

用法示例（在 stereo_center/ 目录下）：
    conda run -n waft python scripts/run_depth_video.py \
        --video ../dataset/xxx/output.mp4 --calib ../dataset/xxx/calibration.json \
        --scale 0.5 --batch-size 4 --outdir outputs/depth_video

- 深度值恒为米制：depth = fx * baseline / disparity。
- 色阶为固定对数米制映射（0.3~20m，可调）：场景深度范围大时线性色阶
  顾此失彼，对数映射让近场/背景都有可分辨色带，且跨帧同深度同色。
- 时间平滑默认关闭（逐帧深度）：时间滤波虽能压抖动，但会引入拖影/重影。
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, softsplat, stereo_backend  # noqa: E402
from stereo_center.pipeline import photometric_align_right  # noqa: E402
from stereo_center.guided_filter import guided_filter  # noqa: E402
from stereo_center.raft_flow import flow_between, load_raft  # noqa: E402
from stereo_center.visualize import (  # noqa: E402
    colorize_depth_log,
    make_depth_colorbar_log,
)


def resolve_weights_dir(explicit: str | None, backend: str) -> Path:
    if explicit:
        return Path(explicit)
    env_map = {"waft": "WAFT_WEIGHTS_DIR", "s2m2": "S2M2_WEIGHTS_DIR", "las2": "LAS2_WEIGHTS_DIR"}
    env = env_map[backend]
    if env in os.environ:
        return Path(os.environ[env])
    repo_root = PROJECT_ROOT.parent
    subs = ["las2", "pretrain_weights"] if backend == "las2" else (["waft"] if backend == "waft" else ["pretrain_weights"])
    for sub in subs:
        for c in (repo_root / "weights" / sub, PROJECT_ROOT / "weights" / sub):
            if c.exists():
                return c
    # 兜底：找目录下含对应权重文件的位置
    fname = f"LAS2_{'M'}.pth" if backend == "las2" else None
    for c in (repo_root / "weights", PROJECT_ROOT / "weights"):
        if c.exists():
            if backend != "las2" or any(c.rglob(fname)):
                return c
    raise FileNotFoundError(f"未找到权重目录（{backend}），请用 --weights 或环境变量 {env}")


def warp_with_flow(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """src: (1,1,H,W)；flow: (1,2,H,W) 像素位移 -> 采样结果 (1,1,H,W)。"""
    B, C, H, W = src.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=src.device),
        torch.arange(W, device=src.device),
        indexing="ij",
    )
    gx = ((xx.float() + flow[:, 0]) / max(W - 1, 1) * 2 - 1).clamp(-1, 1)
    gy = ((yy.float() + flow[:, 1]) / max(H - 1, 1) * 2 - 1).clamp(-1, 1)
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(
        src, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )


def process_batch(
    model,
    bgr_pairs: list,
    rect: dict,
    fx: float,
    baseline: float,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """校正后 BGR 对列表 → GPU 批量推理 + 融合 + 遮挡填充。

    Returns:
        dep_b: (B, 1, H, W) float32 GPU 中心深度（米，已填充）；
        valid_b: (B, 1, H, W) bool GPU 有效掩码；
        rgb_b: (B, 3, H, W) float32 GPU 中心 RGB（0-255）。
    """
    B = len(bgr_pairs)
    H, W = bgr_pairs[0][0].shape[:2]
    left_t = torch.zeros(B, 3, H, W)
    right_t = torch.zeros(B, 3, H, W)
    for b, (rL, rR) in enumerate(bgr_pairs):
        left_t[b] = torch.from_numpy(cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_t[b] = torch.from_numpy(cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

    if args.stereo_backend == "waft":
        from stereo_center import waft_inference  # noqa: E402  # 懒加载：仅 waft 路径需要 peft 等

        if args.bi:
            dL, dR, occL, occR, confL, confR, _ = waft_inference.run_stereo_matching_bi_batch(
                model, left_t, right_t, args.device,
                hiera=args.hiera, conf_mode=args.conf, occ_mode=args.occ,
            )
        else:
            dL, occL, confL, _ = waft_inference.run_stereo_matching(
                model, left_t, right_t, args.device,
                hiera=args.hiera, conf_mode="ones", occ_mode="visibility",
            )
            dR = occR = confR = None
    else:
        if args.bi:
            dL, dR, occL, occR, confL, confR, _ = stereo_backend.run_bi_batch(
                args.stereo_backend, model, left_t, right_t, args.device,
                max_disp=args.max_disp, conf_mode=args.conf, occ_mode=args.occ,
            )
        else:
            dL, occL, confL, _ = stereo_backend.run(
                args.stereo_backend, model, left_t, right_t, args.device,
                max_disp=args.max_disp, conf_mode=args.conf, occ_mode=args.occ,
            )
            dR = occR = confR = None
    if args.guided_filter:
        from stereo_center.guided_filter import guided_filter_batch  # noqa: E402

        dL, dR = guided_filter_batch(
            bgr_pairs, dL.numpy(), dR.numpy() if dR is not None else None,
            args.gf_radius, args.gf_eps,
        )
        dL = torch.from_numpy(dL)
        dR = torch.from_numpy(dR) if dR is not None else None
    dev = args.device
    # 保持逐帧光度校正，但将中心视角融合本身一次性按 B 帧执行，
    # 避免每帧重复创建投影网格、权重和 z-buffer 中间张量。
    right_f_cpu = torch.stack([
        torch.from_numpy(
            cv2.cvtColor(
                photometric_align_right(rL_bgr, rR_bgr), cv2.COLOR_BGR2RGB
            )
        ).permute(2, 0, 1).float()
        for rL_bgr, rR_bgr in bgr_pairs
    ])
    left_f = left_t.to(dev)
    right_f = right_f_cpu.to(dev)
    dl = dL.unsqueeze(1).to(dev)
    cl = confL.unsqueeze(1).to(dev)
    ol = occL.unsqueeze(1).to(dev)
    if dR is not None:
        dr = dR.unsqueeze(1).to(dev)
        cr = confR.unsqueeze(1).to(dev)
        orr = occR.unsqueeze(1).to(dev)
    else:
        dr = cr = orr = None
    rgb_b, dep_b, valid_b = softsplat.center_view(
        left_f, right_f, dl, cl, ol, fx=fx, baseline=baseline,
        disp_right=dr, conf_right=cr, occ_right=orr,
        edge_k=1.5, blend="softz", weight_mode="expdecay", weight_k=4.0,
        median_k=args.median_k,
        depth_z=bool(args.depth_z), depth_z_thresh=0.05, depth_z_power=2.0,
        color_tol=args.color_tol,
    )
    rgb_b, dep_b, valid_b = softsplat.fill_disocclusion_torch(rgb_b, dep_b, valid_b)
    if args.depth_gf:
        # 中心 RGB 引导滤波：深度边缘对齐到图像边缘，提升锐度
        dev = args.device
        for b in range(B):
            c_rgb = rgb_b[b].permute(1, 2, 0).cpu().numpy()
            c_gray = cv2.cvtColor(c_rgb, cv2.COLOR_RGB2GRAY)
            dep_np = dep_b[b, 0].cpu().numpy()
            q = guided_filter(c_gray, dep_np, args.depth_gf_radius, args.depth_gf_eps)
            if args.depth_unsharp > 0:
                # 边缘保留 unsharp：仅在有图像边缘处增强
                q = dep_np + args.depth_unsharp * (dep_np - q)
            dep_b[b, 0] = torch.from_numpy(q).to(dev)
    return dep_b, valid_b, rgb_b


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
    parser.add_argument("--stereo-backend", type=str, default="waft", choices=["waft", "s2m2", "las2"])
    parser.add_argument("--model-type", type=str, default="DAv2L-5")
    parser.add_argument("--max-disp", type=int, default=192, help="LAS2 最大视差（默认 192）")
    parser.add_argument("--las-root", type=str, default=None, help="LiteAnyStereo 仓库根目录（LAS2）")
    parser.add_argument("--waft-iters", type=int, default=None, help="WAFT 迭代轮数（默认取配置 4；2/3 可提速）")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hiera", type=str, default="auto", choices=["auto", "direct", "hiera"])
    parser.add_argument("--bi", type=int, default=1, choices=[0, 1], help="双向视差（1=开，0=仅左方向，速度快约 2 倍）")
    parser.add_argument("--conf", type=str, default="lr", choices=["info", "ones", "lr"])
    parser.add_argument("--occ", type=str, default="lr", choices=["lr", "visibility"])
    parser.add_argument("--color-tol", type=float, default=15.0, help="softz 颜色冲突阈值")
    parser.add_argument("--median-k", type=int, default=0, help="融合前视差中值滤波核（奇数，0/1 关闭；3 可压抖动）")
    parser.add_argument("--guided-filter", type=int, default=0, choices=[0, 1], help="RGB 引导滤波锐化视差")
    parser.add_argument("--gf-radius", type=int, default=8, help="引导滤波窗口半径")
    parser.add_argument("--gf-eps", type=float, default=300.0, help="引导滤波正则项（0-255 灰度尺度）")
    parser.add_argument("--depth-gf", type=int, default=0, choices=[0, 1], help="融合后中心 RGB 引导滤波锐化深度")
    parser.add_argument("--depth-gf-radius", type=int, default=6, help="深度引导滤波窗口半径")
    parser.add_argument("--depth-gf-eps", type=float, default=200.0, help="深度引导滤波正则项")
    parser.add_argument("--depth-unsharp", type=float, default=0.0, help="深度边缘保留 unsharp 强度（0=关）")
    parser.add_argument("--temporal-raft", type=int, default=0, choices=[0, 1], help="RAFT 光流 + 一致性门控时间稳定（防闪）")
    parser.add_argument("--temporal-alpha", type=float, default=0.6, help="RAFT 时间 EMA 融合系数")
    parser.add_argument("--temporal-photo-tol", type=float, default=25.0, help="光度一致性门控容差（灰度差）")
    parser.add_argument("--temporal-depth-tol", type=float, default=0.3, help="深度一致性门控绝对容差（米）")
    parser.add_argument("--temporal-depth-rel", type=float, default=0.15, help="深度一致性门控相对容差")
    parser.add_argument("--raft-weights", type=str, default=None, help="raft-things.pth 路径（--temporal-raft 时必填）")
    parser.add_argument("--raft-root", type=str, default=None, help="RAFT 仓库代码根目录")
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
        "--temporal-median", type=int, default=1,
        help="时间维中值滤波窗口（奇数，1=关闭；时间平滑易引入拖影，默认关闭）",
    )
    parser.add_argument(
        "--temporal-ema", type=float, default=1.0,
        help="时间维 EMA：1=关闭（默认）；0=光流运动补偿（实验性，有重影风险）；"
        ">0 且 <1=固定系数",
    )
    parser.add_argument(
        "--flow-alpha", type=float, default=0.2,
        help="光流 EMA 融合系数（仅 --temporal-ema 0 时生效）",
    )
    parser.add_argument(
        "--spatial-median", type=int, default=1,
        help="逐帧深度空间 3x3 中值滤波核（奇数，1=关闭；默认关闭）",
    )
    parser.add_argument("--save-frames-every", type=int, default=50, help="每隔 N 帧存一张深度 PNG（0=不存）")
    parser.add_argument("--save-depth-npy", type=int, default=0, help="每帧额外保存米制深度 npy（供后处理时间平滑）")
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
    model = stereo_backend.load(
        backend, args.model_type, str(weights_dir), args.device,
        num_refine=3, max_disp=args.max_disp, las_root=args.las_root,
        iters=args.waft_iters,
    )
    raft = None
    if args.temporal_raft:
        raft_weights = (
            Path(args.raft_weights).expanduser().resolve()
            if args.raft_weights
            else Path.home() / "BothEyesDepth/stereoanyvideo/third_party/RAFT/models/raft-things.pth"
        )
        raft = load_raft(raft_weights, args.raft_root, args.device)

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
    prev_left_t = None
    prev_depth_t = None
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
        dep_b, valid_b, _rgb_b = process_batch(model, bgr_pairs, rect, fx, baseline, args)
        t_stereo_fusion_fill += time.time() - t0

        for b in range(B):
            rL_bgr, rR_bgr = bgr_pairs[b]
            d_cur = dep_b[b].unsqueeze(0)  # (1,1,H,W) GPU
            v_cur = valid_b[b].unsqueeze(0)
            if win > 1:
                depth_tbuf.append(d_cur)
                if len(depth_tbuf) == win:
                    d_cur = torch.median(torch.stack(list(depth_tbuf)), dim=0).values
            if args.temporal_raft and raft is not None:
                # RAFT 光流 + 光度/深度一致性门控的时间 EMA（防闪不拖影）
                cur_left_t = (
                    torch.from_numpy(cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2RGB))
                    .permute(2, 0, 1).float().unsqueeze(0).to(args.device)
                )
                if prev_left_t is not None and prev_depth_t is not None:
                    flow = flow_between(raft, prev_left_t, cur_left_t)
                    prev_w = warp_with_flow(prev_depth_t, flow)
                    prev_w_gray = warp_with_flow(
                        prev_left_t.mean(dim=1, keepdim=True), flow
                    )
                    cur_gray_t = cur_left_t.mean(dim=1, keepdim=True)
                    mae = (prev_w_gray - cur_gray_t).abs()
                    g_photo = torch.clamp(
                        1.0 - mae / max(args.temporal_photo_tol, 1e-3), 0.0, 1.0
                    )
                    d_diff = (prev_w - d_cur).abs()
                    g_depth = (
                        d_diff
                        <= torch.maximum(
                            torch.tensor(args.temporal_depth_tol, device=args.device),
                            args.temporal_depth_rel * d_cur,
                        )
                    ).float()
                    g = g_photo * g_depth
                    g = F.avg_pool2d(g, 7, stride=1, padding=3)
                    if os.environ.get("SAV_DEBUG_GATE"):
                        print(
                            f"[gate] frame={frame_idx + b} g_photo={g_photo.mean().item():.3f} "
                            f"g_depth={g_depth.mean().item():.3f} g={g.mean().item():.3f}",
                            flush=True,
                        )
                    d_cur = g * (
                        args.temporal_alpha * prev_w
                        + (1.0 - args.temporal_alpha) * d_cur
                    ) + (1.0 - g) * d_cur
                prev_left_t = cur_left_t
                prev_depth_t = d_cur.detach()
            dep_np = d_cur[0, 0].cpu().numpy()
            valid_np = v_cur[0, 0].cpu().numpy().astype(bool)
            if args.spatial_median > 1:
                dep_np = cv2.medianBlur(dep_np, args.spatial_median)
            # 时间 EMA：默认用光流把上一帧深度 warp 到当前帧再融合（运动补偿）
            if args.temporal_ema != 1.0:
                cur_gray = cv2.cvtColor(rL_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev_depth is not None and prev_gray is not None:
                    if args.temporal_ema > 0.0:
                        alpha = args.temporal_ema
                        warped = prev_depth
                    else:
                        alpha = args.flow_alpha
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, cur_gray, None, **flow_params
                        )
                        yy, xx = np.mgrid[0 : dep_np.shape[0], 0 : dep_np.shape[1]]
                        # Farneback(prev,cur) 的 flow 把 cur 坐标映射回 prev，
                        # 因此 warp prev 深度到 cur 坐标要加 flow
                        mapx = (xx + flow[..., 0]).astype(np.float32)
                        mapy = (yy + flow[..., 1]).astype(np.float32)
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
            if args.save_depth_npy:
                np.save(
                    str(outdir / f"depth_{frame_idx + b:06d}.npy"),
                    dep_np.astype(np.float32),
                )
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
        "waft_iters": args.waft_iters,
        "bidirectional": bool(args.bi),
        "max_disp": args.max_disp if backend == "las2" else None,
        "start_frame": args.start_frame,
        "end_frame": frame_idx,
        "n_frames": processed,
        "fps": fps,
        "size": [W, H],
        "total_seconds": round(total_s, 2),
        "avg_seconds_per_frame": round(total_s / max(processed, 1), 4),
        "color_tol": args.color_tol,
        "median_k": args.median_k,
        "guided_filter": bool(args.guided_filter),
        "gf_radius": args.gf_radius,
        "gf_eps": args.gf_eps,
        "depth_gf": bool(args.depth_gf),
        "depth_gf_radius": args.depth_gf_radius,
        "depth_gf_eps": args.depth_gf_eps,
        "depth_unsharp": args.depth_unsharp,
        "temporal_raft": bool(args.temporal_raft),
        "temporal_alpha": args.temporal_alpha,
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
