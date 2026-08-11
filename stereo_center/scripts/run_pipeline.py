#!/usr/bin/env python
"""在单帧双目视频帧上运行 S²M² + SoftSplat 最小管线。

用法示例（在 stereo_center/ 目录下）：
    ../.venv/bin/python scripts/run_pipeline.py \
        --video ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/output.mp4 \
        --calib ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/calibration.json \
        --frame 60 --scale 0.5 --outdir outputs/run_1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent  # 仓库根目录（clone 后的 CenterDepth/）
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/s2m2/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereo_center import calib, pipeline, pointcloud, stereo_backend  # noqa: E402
from stereo_center.visualize import colorize_depth, colorize_map, make_overview  # noqa: E402


def resolve_weights_dir(explicit: str | None, backend: str = "s2m2") -> Path:
    """权重目录解析：--weights > 后端环境变量 > 仓库根 weights/<backend>/ > 旧路径。"""
    if explicit:
        return Path(explicit)
    if backend == "waft":
        env = os.environ.get("WAFT_WEIGHTS_DIR")
        subdir = "waft"
    else:
        env = os.environ.get("S2M2_WEIGHTS_DIR")
        subdir = "pretrain_weights"
    if env:
        return Path(env)
    candidates = [REPO_ROOT / "weights" / subdir, PROJECT_ROOT / "weights" / subdir]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"未找到权重目录（后端 {backend}）。请用 --weights 指定，或设置环境变量"
        f" {env or 'S2M2_WEIGHTS_DIR'}，或将权重放到 {candidates[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="S²M² + SoftSplat 最小管线")
    parser.add_argument("--video", type=str, required=True, help="双目视频 (3840x1200)")
    parser.add_argument("--calib", type=str, required=True, help="calibration.json")
    parser.add_argument("--frame", type=int, default=60, help="视频帧索引")
    parser.add_argument("--scale", type=float, default=0.5, help="校正输出缩放（CPU 建议 0.5）")
    parser.add_argument(
        "--stereo-backend", type=str, default="waft", choices=["s2m2", "waft"],
        help="立体匹配后端（默认 waft；s2m2 可回退）",
    )
    parser.add_argument(
        "--model-type", type=str, default="DAv2L-5",
        choices=["S", "M", "L", "XL", "DAv2S-4", "DAv2B-4", "DAv2L-5"],
        help="模型类型：s2m2 用 S/M/L/XL；waft 用 DAv2S-4/DAv2B-4/DAv2L-5",
    )
    parser.add_argument("--num-refine", type=int, default=3)
    parser.add_argument(
        "--weights", type=str, default=None,
        help="权重目录（默认按后端解析：waft->weights/waft 或 $WAFT_WEIGHTS_DIR）",
    )
    parser.add_argument(
        "--waft-mode", type=str, default="auto", choices=["auto", "direct", "hiera"],
        help="WAFT 推理模式：auto 在 >1080 时用 0.5->1.0 分层",
    )
    parser.add_argument(
        "--waft-conf", type=str, default="lr", choices=["info", "ones", "lr"],
        help="WAFT 置信度：lr=平滑左右一致性（默认，可解释）；info=官方 uncertainty 映射；ones=常量 1",
    )
    parser.add_argument(
        "--waft-occ", type=str, default="lr", choices=["lr", "visibility"],
        help="WAFT 遮挡掩码：lr=左右一致性（双向）；visibility=仅可见性（单向）",
    )
    parser.add_argument(
        "--fusion", type=str, default="improved", choices=["baseline", "improved"],
        help="中心视角融合：baseline=旧软平均（单侧视差、无改进）；improved=低成本融合改进",
    )
    parser.add_argument(
        "--fusion-bi", type=int, default=None, choices=[0, 1],
        help="覆盖融合选项：双向视差（improved 默认 1）",
    )
    parser.add_argument(
        "--fusion-photometric", type=int, default=None, choices=[0, 1],
        help="覆盖融合选项：光度校正（improved 默认 1）",
    )
    parser.add_argument(
        "--fusion-edge-k", type=float, default=None,
        help="覆盖融合选项：边缘感知权重系数，0=关（improved 默认 1.5）",
    )
    parser.add_argument(
        "--fusion-median-k", type=int, default=None,
        help="覆盖融合选项：视差中值滤波核，0/1=关（improved 默认 0）",
    )
    parser.add_argument(
        "--fusion-fill", type=int, default=None, choices=[0, 1],
        help="覆盖融合选项：背景深度遮挡填充（improved 默认 1）",
    )
    parser.add_argument(
        "--fusion-blend", type=str, default=None, choices=["softavg", "gate", "hybrid", "conflict"],
        help="覆盖融合选项：softavg=软平均；gate=深度一致性门控；hybrid=RGB 软平均+Depth 门控；conflict=软平均+冲突抑制（improved 默认 conflict）",
    )
    parser.add_argument(
        "--fusion-color-tol", type=float, default=None,
        help="覆盖融合选项：conflict 模式颜色冲突阈值（0-255，improved 默认 25）",
    )
    parser.add_argument("--no-pointcloud", action="store_true", help="不输出 3D 点云")
    parser.add_argument("--max-points", type=int, default=300_000, help="点云随机下采样上限")
    parser.add_argument("--stride", type=int, default=2, help="点云深度图采样步长（>1 减少点数）")
    parser.add_argument("--z-max", type=float, default=10.0, help="3D 点云可视化深度截断（米）")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/run_1"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    backend = args.stereo_backend
    if backend == "s2m2" and args.model_type not in ("S", "M", "L", "XL"):
        raise ValueError(f"s2m2 后端不支持模型类型 {args.model_type}（可选 S/M/L/XL）")
    if backend == "waft" and args.model_type not in ("DAv2S-4", "DAv2B-4", "DAv2L-5"):
        raise ValueError(
            f"waft 后端不支持模型类型 {args.model_type}（可选 DAv2S-4/DAv2B-4/DAv2L-5）"
        )

    if args.fusion == "baseline":
        fusion = {
            "bi": False,
            "photometric": False,
            "edge_k": 0.0,
            "median_k": 0,
            "fill_holes": False,
            "blend": "softavg",
            "color_tol": 25.0,
        }
    else:
        fusion = dict(pipeline.DEFAULT_FUSION)
    if args.fusion_bi is not None:
        fusion["bi"] = bool(args.fusion_bi)
    if args.fusion_photometric is not None:
        fusion["photometric"] = bool(args.fusion_photometric)
    if args.fusion_edge_k is not None:
        fusion["edge_k"] = args.fusion_edge_k
    if args.fusion_median_k is not None:
        fusion["median_k"] = args.fusion_median_k
    if args.fusion_fill is not None:
        fusion["fill_holes"] = bool(args.fusion_fill)
    if args.fusion_blend is not None:
        fusion["blend"] = args.fusion_blend
    if args.fusion_color_tol is not None:
        fusion["color_tol"] = args.fusion_color_tol
    if fusion["blend"] not in ("softavg", "gate", "hybrid", "conflict"):
        raise ValueError(
            f"未知融合模式: {fusion['blend']}（可选 softavg/gate/hybrid/conflict）"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读帧并切分左右
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"读取视频第 {args.frame} 帧失败: {args.video}")
    h, w = img.shape[:2]
    left_bgr, right_bgr = img[:, : w // 2], img[:, w // 2 :]
    print(f"[video] 帧 {args.frame}: {w}x{h} -> 左右各 {left_bgr.shape[1]}x{left_bgr.shape[0]}")

    # 2) 权重 + 标定 + 模型
    weights_dir = resolve_weights_dir(args.weights, backend)
    print(f"[weights] 权重目录: {weights_dir}")
    cal = calib.load_vdego_calibration(args.calib)
    print(f"[calib] baseline={cal['baseline']:.4f} m, 分辨率={cal['resolution']}")
    model = stereo_backend.load(
        backend, args.model_type, str(weights_dir), args.device,
        num_refine=args.num_refine,
    )
    backend_kwargs = {}
    if backend == "waft":
        backend_kwargs = {
            "hiera": args.waft_mode,
            "conf_mode": args.waft_conf,
            "occ_mode": args.waft_occ,
        }

    # 3) 管线
    res = pipeline.process_stereo_pair(
        left_bgr, right_bgr, cal, model,
        device=args.device, scale=args.scale, backend=backend,
        backend_kwargs=backend_kwargs,
        fusion=fusion,
    )
    print(f"[{backend}] 单帧推理耗时 {res.elapsed_s2m2:.1f} s")
    conf = res.conf[100:-100, 100:-100] if res.conf.shape[0] > 200 else res.conf
    print(f"[{backend}] 平均置信度: {conf.mean():.3f}")

    # 4) 保存产物
    cv2.imwrite(str(outdir / "rect_left.png"), res.rect_left)
    cv2.imwrite(str(outdir / "rect_right.png"), res.rect_right)
    np.save(str(outdir / "disparity.npy"), res.disp)
    if res.disp_right is not None:
        np.save(str(outdir / "disp_right.npy"), res.disp_right)
    np.save(str(outdir / "occlusion.npy"), res.occ)
    np.save(str(outdir / "confidence.npy"), res.conf)
    cv2.imwrite(str(outdir / "disparity.png"), colorize_map(res.disp))
    cv2.imwrite(
        str(outdir / "occlusion.png"),
        cv2.normalize(res.occ, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    )
    # 置信度 PNG 用百分位拉伸显示（避免 info 模式双峰分布导致大片黑色）
    conf_p1, conf_p99 = np.percentile(res.conf, 1), np.percentile(res.conf, 99)
    if conf_p99 - conf_p1 < 1e-6:
        conf_p1, conf_p99 = float(res.conf.min()), float(res.conf.max() + 1e-6)
    conf_disp = np.clip((res.conf - conf_p1) / (conf_p99 - conf_p1), 0, 1)
    cv2.imwrite(
        str(outdir / "confidence.png"),
        (conf_disp * 255).astype(np.uint8),
    )
    cv2.imwrite(str(outdir / "center_rgb.png"), res.center_rgb)
    np.save(str(outdir / "center_depth.npy"), res.center_depth)
    cv2.imwrite(str(outdir / "center_depth.png"), colorize_depth(res.center_depth, res.center_valid))

    # 5.5) 3D 点云（左+右视差重建，可选）
    num_points = 0
    if not args.no_pointcloud:
        depthL = res.fx * res.baseline / np.maximum(res.disp, 0.5)
        ptsL, colL = pointcloud.depth_to_pointcloud(
            res.rect_left, depthL, res.fx, res.fy, res.cx, res.cy,
            max_points=args.max_points, stride=args.stride,
        )
        pts, col = ptsL, colL
        if res.disp_right is not None:
            depthR = res.fx * res.baseline / np.maximum(res.disp_right, 0.5)
            ptsR, colR = pointcloud.depth_to_pointcloud(
                res.rect_right, depthR, res.fx, res.fy, res.cx, res.cy,
                max_points=args.max_points, stride=args.stride,
            )
            ptsR = pointcloud.transform_right_to_left(ptsR, res.baseline)
            pts = np.concatenate([pts, ptsR], axis=0)
            col = np.concatenate([col, colR], axis=0)
        num_points = int(len(pts))
        np.savez(str(outdir / "pointcloud.npz"), points=pts, colors=col)
        pointcloud.save_ply(pts, col, outdir / "pointcloud.ply")
        try:
            pointcloud.visualize_pointcloud(
                pts, col, outdir / "pointcloud_3d.png", z_max=args.z_max
            )
        except Exception as exc:  # matplotlib 缺失时不影响主流程
            print(f"[pointcloud] 3D 可视化跳过（{exc}）")
        print(f"[pointcloud] {num_points} 点已保存 (ply/npz/png)")

    # 5) 总览图：左校正 | 中心RGB | 右校正 / 视差 | 中心深度 | 置信度
    overview = make_overview(
        res.rect_left,
        res.center_rgb,
        res.rect_right,
        res.disp,
        res.center_depth,
        res.center_valid,
        res.conf,
    )
    cv2.imwrite(str(outdir / "overview.png"), overview)

    valid_frac = res.center_valid.mean()
    d_valid = res.center_depth[res.center_valid]
    print(
        f"[center] 有效像素占比 {valid_frac:.1%}，深度范围 "
        f"{d_valid.min():.2f}~{d_valid.max():.2f} m (均值 {d_valid.mean():.2f})"
    )

    # 6) 统计信息（JSON）
    stats = {
        "stereo_backend": backend,
        "model_type": args.model_type,
        "num_refine": args.num_refine,
        "device": args.device,
        "frame": args.frame,
        "scale": args.scale,
        "fx": round(res.fx, 3),
        "baseline": round(res.baseline, 4),
        "stereo_inference_seconds": round(res.elapsed_s2m2, 3),
        "s2m2_inference_seconds": round(res.elapsed_s2m2, 3),
        "mean_confidence": round(float(conf.mean()), 4),
        "valid_pixel_fraction": round(float(valid_frac), 4),
        "depth_min_m": round(float(d_valid.min()), 4),
        "depth_max_m": round(float(d_valid.max()), 4),
        "depth_mean_m": round(float(d_valid.mean()), 4),
        "depth_median_m": round(float(np.median(d_valid)), 4),
        "fusion_ambiguity": round(res.fusion_ambiguity, 3),
        "fusion_single_fraction": round(res.fusion_single_fraction, 5),
        "num_points": num_points,
    }
    if backend == "waft":
        stats.update(
            {
                "waft_mode": args.waft_mode,
                "waft_conf": args.waft_conf,
                "waft_occ": args.waft_occ,
            }
        )
    stats.update(
        {
            "fusion": args.fusion,
            "fusion_bi": int(fusion["bi"]),
            "fusion_photometric": int(fusion["photometric"]),
            "fusion_edge_k": fusion["edge_k"],
            "fusion_median_k": fusion["median_k"],
            "fusion_fill_holes": int(fusion["fill_holes"]),
            "fusion_blend": fusion["blend"],
            "fusion_color_tol": fusion["color_tol"],
        }
    )
    with open(outdir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[stats] 已写入 {outdir / 'stats.json'}")
    print(f"[out] 结果已保存到 {outdir}")


if __name__ == "__main__":
    main()
