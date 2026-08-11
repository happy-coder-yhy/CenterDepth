"""组合管线：双目帧 -> 鱼眼校正 -> 立体匹配（s2m2|waft）-> SoftSplat -> 中心视角 RGB + Depth。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from . import calib, softsplat, stereo_backend

DEFAULT_FUSION = {
    "bi": True,           # 双向视差（右参考视差直接推理）
    "photometric": True,  # 融合前右图光度对齐到左图
    "edge_k": 1.5,        # 边缘感知权重系数（0=关）
    "median_k": 0,        # 视差中值滤波核（0/1=关；本地消融显示无益，默认关）
    "fill_holes": True,   # 背景深度遮挡填充
    "blend": "softavg",   # softavg / gate
}


@dataclass
class PipelineResult:
    rect_left: np.ndarray  # BGR
    rect_right: np.ndarray  # BGR
    disp: np.ndarray  # (H, W) float32 左视差（像素）
    occ: np.ndarray  # (H, W) float32
    conf: np.ndarray  # (H, W) float32
    center_rgb: np.ndarray  # (H, W, 3) uint8
    center_depth: np.ndarray  # (H, W) float32（米），无效处为 0
    center_valid: np.ndarray  # (H, W) bool
    elapsed_s2m2: float
    fx: float  # 校正后焦距（像素）
    baseline: float  # 基线（米）
    fusion_ambiguity: float  # 融合歧义度：左右 warp 到中心后颜色差异（越低重影越少）
    fusion_single_fraction: float  # 融合前单视图覆盖占比（遮挡/未对齐区域）


def photometric_align_right(
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> np.ndarray:
    """把右图亮度/颜色对齐到左图（每通道 gain/offset），避免融合亮度接缝。"""
    gL = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gR = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (gL > 30) & (gR > 30)
    if not mask.any():
        return right_bgr.copy()
    out = right_bgr.copy()
    for c in range(3):
        lc = left_bgr[:, :, c].astype(np.float32)
        rc = right_bgr[:, :, c].astype(np.float32)
        a = lc[mask].std() / max(rc[mask].std(), 1e-6)
        b = lc[mask].mean() - a * rc[mask].mean()
        out[:, :, c] = np.clip(a * rc + b, 0, 255)
    return out


def process_stereo_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    cal: dict,
    model,
    device: str = "cpu",
    scale: float = 0.5,
    backend: str = "waft",
    backend_kwargs: dict | None = None,
    fusion: dict | None = None,
) -> PipelineResult:
    """处理一帧双目图，输出中心视角 RGB + Depth。"""
    f = dict(DEFAULT_FUSION)
    if fusion:
        f.update(fusion)
    out_size = (
        max(32, int(cal["resolution"][0] * scale)),
        max(32, int(cal["resolution"][1] * scale)),
    )
    rect = calib.compute_rectification_maps(cal, output_size=out_size)
    rL, rR = calib.rectify_pair(left_bgr, right_bgr, rect)

    left_rgb = cv2.cvtColor(rL, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(rR, cv2.COLOR_BGR2RGB)
    left_t = torch.from_numpy(left_rgb).permute(2, 0, 1).float().unsqueeze(0)
    right_t = torch.from_numpy(right_rgb).permute(2, 0, 1).float().unsqueeze(0)

    mod = stereo_backend.get_backend(backend)
    kwargs = dict(backend_kwargs or {})
    disp_right = conf_right = occ_right = None
    if f["bi"] and hasattr(mod, "run_stereo_matching_bi"):
        dL, dR, occL, occR, confL, confR, elapsed = mod.run_stereo_matching_bi(
            model, left_t, right_t, device, **kwargs
        )
        disp, occ, conf = dL, occL, confL
        confR_mean = float(confR.mean())
        if confR_mean >= 0.1:
            disp_right = dR.unsqueeze(0).unsqueeze(0)
            conf_right = confR.unsqueeze(0).unsqueeze(0)
            occ_right = occR.unsqueeze(0).unsqueeze(0)
        else:
            # 右参考输出不可靠（如 s2m2 反向匹配置信度过低）：
            # 回退为"左视差近似右流"（旧行为），避免右视图以错误几何参与融合
            print(
                f"[fusion] 右参考置信度过低 ({confR_mean:.3f})，"
                f"回退为左视差近似右流"
            )
    else:
        disp, occ, conf, elapsed = mod.run_stereo_matching(
            model, left_t, right_t, device, **kwargs
        )

    # 融合用的右图做光度对齐（匹配仍用原始校正图，互不影响）
    rR_fusion = (
        photometric_align_right(rL, rR) if f["photometric"] else rR
    )
    right_rgb_f = cv2.cvtColor(rR_fusion, cv2.COLOR_BGR2RGB)
    right_t_f = (
        torch.from_numpy(right_rgb_f).permute(2, 0, 1).float().unsqueeze(0)
    )

    center_rgb, center_depth, valid, warp_extra = softsplat.center_view(
        left_t, right_t_f,
        disp.unsqueeze(0).unsqueeze(0),
        conf.unsqueeze(0).unsqueeze(0),
        occ.unsqueeze(0).unsqueeze(0),
        fx=rect["fx"],
        baseline=rect["baseline"],
        disp_right=disp_right,
        conf_right=conf_right,
        occ_right=occ_right,
        edge_k=f["edge_k"],
        median_k=f["median_k"],
        blend=f["blend"],
        return_warped=True,
    )

    # 融合歧义指标（在遮挡填充前计算）：
    # 两视图深度一致的像素上，左右 warp 颜色的平均差异越小，重影越少
    eps = 1e-6
    both = (warp_extra["norm_l"] > eps) & (warp_extra["norm_r"] > eps)
    dep_l, dep_r = warp_extra["dep_l"], warp_extra["dep_r"]
    agree = (dep_l - dep_r).abs() <= 0.15 * torch.minimum(dep_l, dep_r).clamp_min(0.1)
    sel = both & agree
    if bool(sel.any()):
        diff = (warp_extra["rgb_l"] - warp_extra["rgb_r"]).abs().mean(dim=1)  # (1,H,W)
        fusion_ambiguity = float(diff[sel.squeeze(1)].mean())
    else:
        fusion_ambiguity = 0.0
    fusion_single_fraction = float((valid & ~both).float().mean())

    center_rgb_np = (
        center_rgb[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()
    )
    center_depth_np = center_depth[0, 0].numpy()
    valid_np = valid[0, 0].numpy().astype(bool)
    if f["fill_holes"]:
        center_rgb_np, center_depth_np, valid_np = softsplat.fill_disocclusion(
            center_rgb_np, center_depth_np, valid_np
        )

    return PipelineResult(
        rect_left=rL,
        rect_right=rR,
        disp=disp.numpy(),
        occ=occ.numpy(),
        conf=conf.numpy(),
        center_rgb=cv2.cvtColor(center_rgb_np, cv2.COLOR_RGB2BGR),
        center_depth=center_depth_np,
        center_valid=valid_np,
        elapsed_s2m2=elapsed,
        fx=rect["fx"],
        baseline=rect["baseline"],
        fusion_ambiguity=fusion_ambiguity,
        fusion_single_fraction=fusion_single_fraction,
    )
