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
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent  # 仓库根目录（clone 后的 CenterDepth/）
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stereo_center import calib, pipeline, s2m2_inference  # noqa: E402
from stereo_center.visualize import colorize_depth, colorize_map, make_overview  # noqa: E402


def resolve_weights_dir(explicit: str | None) -> Path:
    """权重目录解析：--weights > $S2M2_WEIGHTS_DIR > 仓库根 weights/ > 旧路径。"""
    if explicit:
        return Path(explicit)
    env = os.environ.get("S2M2_WEIGHTS_DIR")
    if env:
        return Path(env)
    candidates = [
        REPO_ROOT / "weights/pretrain_weights",  # 新约定：仓库根目录
        PROJECT_ROOT / "weights/pretrain_weights",  # 旧约定：stereo_center/ 下
    ]
    for c in candidates:
        if (c / "CH128NTR1.pth").exists() or c.exists():
            return c
    raise FileNotFoundError(
        f"未找到权重目录。请用 --weights 指定，或设置环境变量 S2M2_WEIGHTS_DIR，"
        f"或将权重放到 {candidates[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="S²M² + SoftSplat 最小管线")
    parser.add_argument("--video", type=str, required=True, help="双目视频 (3840x1200)")
    parser.add_argument("--calib", type=str, required=True, help="calibration.json")
    parser.add_argument("--frame", type=int, default=60, help="视频帧索引")
    parser.add_argument("--scale", type=float, default=0.5, help="校正输出缩放（CPU 建议 0.5）")
    parser.add_argument("--model-type", type=str, default="S", choices=["S", "M", "L", "XL"])
    parser.add_argument("--num-refine", type=int, default=3)
    parser.add_argument(
        "--weights", type=str, default=None,
        help="权重目录（默认：仓库根 weights/pretrain_weights 或 $S2M2_WEIGHTS_DIR）",
    )
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs/run_1"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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
    weights_dir = resolve_weights_dir(args.weights)
    print(f"[weights] 权重目录: {weights_dir}")
    cal = calib.load_vdego_calibration(args.calib)
    print(f"[calib] baseline={cal['baseline']:.4f} m, 分辨率={cal['resolution']}")
    model = s2m2_inference.load_s2m2(
        args.model_type, weights_dir, args.num_refine, args.device
    )

    # 3) 管线
    res = pipeline.process_stereo_pair(
        left_bgr, right_bgr, cal, model, device=args.device, scale=args.scale
    )
    print(f"[s2m2] 单帧推理耗时 {res.elapsed_s2m2:.1f} s")
    conf = res.conf[100:-100, 100:-100] if res.conf.shape[0] > 200 else res.conf
    print(f"[s2m2] 平均置信度: {conf.mean():.3f}")

    # 4) 保存产物
    cv2.imwrite(str(outdir / "rect_left.png"), res.rect_left)
    cv2.imwrite(str(outdir / "rect_right.png"), res.rect_right)
    np.save(str(outdir / "disparity.npy"), res.disp)
    np.save(str(outdir / "occlusion.npy"), res.occ)
    np.save(str(outdir / "confidence.npy"), res.conf)
    cv2.imwrite(str(outdir / "disparity.png"), colorize_map(res.disp))
    cv2.imwrite(
        str(outdir / "occlusion.png"),
        cv2.normalize(res.occ, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    )
    cv2.imwrite(
        str(outdir / "confidence.png"),
        cv2.normalize(res.conf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    )
    cv2.imwrite(str(outdir / "center_rgb.png"), res.center_rgb)
    np.save(str(outdir / "center_depth.npy"), res.center_depth)
    cv2.imwrite(str(outdir / "center_depth.png"), colorize_depth(res.center_depth, res.center_valid))

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
    print(f"[out] 结果已保存到 {outdir}")


if __name__ == "__main__":
    main()
