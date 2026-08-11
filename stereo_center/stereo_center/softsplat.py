"""纯 PyTorch 的 Softmax Splatting（前向投影），CPU / GPU 通用。

S²M² 官方推荐的 sniklaus/softmax-splatting 是 CUDA 算子；本工程在 Mac
（Apple Silicon，无 CUDA）上做可行性验证，因此用 scatter 实现等价的前向
软投影：每个源像素以 exp(weight) 加权投到目标像素，再做归一化。

面向中心视角合成的低成本融合改进（全部可开关）：
- 双向视差：右参考视差由右图直接推理，替代"从左图近似推导右流"；
- 光度校正：融合前把右图每通道 gain/offset 对齐到左图（在 pipeline 层）；
- 边缘感知权重：深度边缘（|∇d| 大）的源像素降权，减少跨边界混色重影；
- 视差中值滤波：消除视差斑点噪声；
- 深度一致性门控（可选）：两视图深度不一致时只保留更近的一侧；
- 背景深度遮挡填充：DIBR 孔洞填充，孔洞从"深度最大（背景）的邻居"逐层生长。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def softmax_splatting(
    feature: torch.Tensor,
    flow: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """softmax 前向投影。

    Args:
        feature: (B, C, H, W) 待投影特征（RGB / 深度）。
        flow: (B, 2, H, W)，flow[:, 0]=u(水平), flow[:, 1]=v(垂直)，
              目标位置 = 源位置 + flow。
        weight: (B, 1, H, W) 每个源像素的权重（如置信度）。

    Returns:
        warped: (B, C, H, W) 归一化投影结果；
        norm: (B, 1, H, W) 各目标像素收到的 exp(weight) 之和（用于多视图融合）；
        valid: (B, 1, H, W) 至少收到一个投影的掩码。
    """
    B, C, H, W = feature.shape
    wgt = torch.exp(weight)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=feature.device),
        torch.arange(W, device=feature.device),
        indexing="ij",
    )
    tx = (xx.unsqueeze(0) + flow[:, 0]).round().long()  # (B, H, W)
    ty = (yy.unsqueeze(0) + flow[:, 1]).round().long()
    valid = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
    idx = (ty * W + tx).reshape(B, -1)
    v = valid.reshape(B, -1)

    feat_w = (feature * wgt).reshape(B, C, -1)
    wgt_flat = wgt.reshape(B, 1, -1)
    out = torch.zeros(B, C, H * W, dtype=feature.dtype, device=feature.device)
    norm = torch.zeros(B, 1, H * W, dtype=feature.dtype, device=feature.device)
    for b in range(B):
        idx_b = idx[b][v[b]]
        out[b].index_add_(1, idx_b, feat_w[b][:, v[b]])
        norm[b].index_add_(1, idx_b, wgt_flat[b][:, v[b]])

    out = out.reshape(B, C, H, W)
    norm = norm.reshape(B, 1, H, W)
    warped = out / norm.clamp_min(torch.finfo(feature.dtype).tiny)
    return warped, norm, valid.unsqueeze(1)


def center_view(
    left_rgb: torch.Tensor,
    right_rgb: torch.Tensor,
    disp_left: torch.Tensor,
    conf_left: torch.Tensor,
    occ_left: torch.Tensor,
    fx: float,
    baseline: float,
    disp_right: torch.Tensor | None = None,
    conf_right: torch.Tensor | None = None,
    occ_right: torch.Tensor | None = None,
    edge_k: float = 0.0,
    median_k: int = 0,
    blend: str = "softavg",
    depth_tol: float = 0.15,
    return_warped: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """由校正后双目 + 立体匹配结果合成中心视角 RGB 与深度。

    几何：中心像素 xM = (xL + xR) / 2
      - 左图到中心 flow: u_L = -dL/2（dL 为左参考视差）
      - 右图到中心 flow: u_R = +dR/2（dR 为右参考视差；未提供时用
        左视差前向投影近似）

    Args:
        left_rgb/right_rgb: (1, 3, H, W) 0-255 float tensor（校正后）。
        disp_left: (1, 1, H, W) 左视差 d = xL - xR。
        conf_left/occ_left: (1, 1, H, W) S²M² 置信度 / 遮挡掩码。
        disp_right: (1, 1, H, W) 右参考视差 dR = xR - xL（双向匹配得到）。
        conf_right/occ_right: (1, 1, H, W) 右参考置信度 / 遮挡掩码。
        fx: 校正后焦距（像素）；baseline: 基线（米）。
        edge_k: 边缘感知权重系数（0 关闭；>0 时 |∇d| 大的源像素降权）。
        median_k: 视差中值滤波核（0/1 关闭，建议 3）。
        blend: softavg=置信度加权软平均（默认）；gate=深度一致性门控
               （不一致时只保留更近一侧，默认关闭）。
        depth_tol: blend=gate 时的相对深度容差。

    Returns:
        center_rgb: (1, 3, H, W) 中心 RGB；
        center_depth: (1, 1, H, W) 中心深度（米），无效处为 0；
        valid: (1, 1, H, W) 有效掩码。
    """
    d_l = _median_filter(disp_left.clamp(min=0.0), median_k)
    weight_left = (conf_left * occ_left).clamp(0.0, 1.0)
    if edge_k > 0:
        weight_left = weight_left * _edge_weight(d_l, edge_k)
    zero = torch.zeros_like(d_l)

    # ---- 左视图 -> 中心 ----
    flow_l = torch.cat([-d_l / 2.0, zero], dim=1)
    depth_left = fx * baseline / d_l.clamp_min(0.5)
    rgb_l, norm_l, valid_l = softmax_splatting(left_rgb, flow_l, weight_left)
    dep_l, _, _ = softmax_splatting(depth_left, flow_l, weight_left)

    # ---- 右视图 -> 中心 ----
    if disp_right is not None:
        d_r = _median_filter(disp_right.clamp(min=0.0), median_k)
        weight_right = (conf_right * occ_right).clamp(0.0, 1.0)
        if edge_k > 0:
            weight_right = weight_right * _edge_weight(d_r, edge_k)
        flow_r = torch.cat([d_r / 2.0, zero], dim=1)
        depth_right = fx * baseline / d_r.clamp_min(0.5)
        rgb_r, norm_r, valid_r = softmax_splatting(
            right_rgb, flow_r, weight_right
        )
        dep_r, _, _ = softmax_splatting(depth_right, flow_r, weight_right)
    else:
        # 回退（旧行为）：把 d/2 场从左图坐标前向投影到右图坐标，得到 dR(xR)/2
        flow_l2r = torch.cat([-d_l, zero], dim=1)
        d_half_at_r, _, _ = softmax_splatting(d_l / 2.0, flow_l2r, weight_left)
        flow_r = torch.cat([d_half_at_r, zero], dim=1)
        depth_right = fx * baseline / (2.0 * d_half_at_r).clamp_min(0.5)
        ones = torch.ones_like(weight_left)
        rgb_r, norm_r, valid_r = softmax_splatting(right_rgb, flow_r, ones)
        dep_r, _, _ = softmax_splatting(depth_right, flow_r, ones)

    # ---- 融合 ----
    if blend == "gate":
        both = valid_l & valid_r
        agree = (dep_l - dep_r).abs() <= depth_tol * torch.minimum(dep_l, dep_r).clamp_min(0.1)
        closer_l = dep_l <= dep_r
        keep_l = agree | ~both | closer_l
        keep_r = agree | ~both | (~closer_l)
        w_l = norm_l * keep_l.to(norm_l.dtype)
        w_r = norm_r * keep_r.to(norm_r.dtype)
    else:  # softavg（默认，保持原有行为）
        w_l, w_r = norm_l, norm_r
    wsum = w_l + w_r
    valid = wsum > 1e-6
    center_rgb = (w_l * rgb_l + w_r * rgb_r) / wsum.clamp_min(1e-6)
    center_depth = (w_l * dep_l + w_r * dep_r) / wsum.clamp_min(1e-6)
    center_depth = center_depth * valid.to(center_depth.dtype)
    if return_warped:
        return (
            center_rgb,
            center_depth,
            valid,
            {
                "rgb_l": rgb_l,
                "rgb_r": rgb_r,
                "norm_l": norm_l,
                "norm_r": norm_r,
                "dep_l": dep_l,
                "dep_r": dep_r,
            },
        )
    return center_rgb, center_depth, valid


def _median_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    """滑窗中值滤波（反射填充），用于消除视差斑点噪声。k 为奇数核大小。"""
    if k <= 1:
        return x
    pad = int(k) // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    win = xp.unfold(2, int(k), 1).unfold(3, int(k), 1)  # (B, C, H, W, k, k)
    return win.reshape(*win.shape[:4], int(k) * int(k)).median(dim=-1).values


def _edge_weight(disp: torch.Tensor, k: float) -> torch.Tensor:
    """边缘感知权重：|∇d| 归一化后指数降权（|∇d| 大 = 深度边缘，易混色）。"""
    sobel = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=disp.dtype,
        device=disp.device,
    ).view(1, 1, 3, 3)
    dx = F.conv2d(disp, sobel, padding=1)
    dy = F.conv2d(disp, sobel.transpose(2, 3), padding=1)
    grad = (dx * dx + dy * dy).sqrt()
    flat = grad.reshape(grad.shape[0], -1)
    p95 = flat.quantile(0.95, dim=1).view(-1, 1, 1, 1)
    g = (grad / p95.clamp_min(1e-6)).clamp(0.0, 3.0)
    return torch.exp(-k * g)


def _neighbor(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """返回 a 沿 (dy, dx) 方向平移后的数组（边界补 0，保持形状）。"""
    H, W = a.shape[:2]
    out = np.zeros_like(a)
    sy0, sy1 = max(0, dy), min(H, H + dy)
    ty0, ty1 = max(0, -dy), min(H, H - dy)
    sx0, sx1 = max(0, dx), min(W, W + dx)
    tx0, tx1 = max(0, -dx), min(W, W - dx)
    out[ty0:ty1, tx0:tx1] = a[sy0:sy1, sx0:sx1]
    return out


def fill_disocclusion(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    max_iter: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于背景深度的 DIBR 遮挡填充（低成本孔洞修复）。

    中心视角中既无左图也无右图覆盖的孔洞（遮挡区）通常属于背景；
    这里让孔洞像素逐层从"深度最大（背景）的已填充邻居"生长，
    避免前景颜色/深度混入遮挡区。

    Args:
        rgb: (H, W, 3) float32/uint8 中心 RGB。
        depth: (H, W) float32 中心深度（米），孔洞处为 0。
        valid: (H, W) bool 有效掩码。
        max_iter: 最大生长迭代数（约等于最大孔洞宽度像素数）。

    Returns:
        填充后的 (rgb, depth, valid)（rgb 保持原 dtype）。
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return rgb, depth, valid
    H, W = depth.shape
    col = rgb.astype(np.float32)
    dep = depth.copy()
    filled = valid.copy()
    rgb3 = col.ndim == 3

    for _ in range(max_iter):
        any_nb = np.zeros((H, W), dtype=bool)
        best_d = np.full((H, W), -np.inf, dtype=np.float32)
        best_c = np.zeros((H, W, 3), dtype=np.float32) if rgb3 else None
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d_nb = _neighbor(dep, dy, dx)
            v_nb = _neighbor(filled.astype(np.float32), dy, dx) > 0.5
            any_nb |= v_nb
            sel = v_nb & (d_nb >= best_d)
            best_d = np.where(sel, d_nb, best_d)
            if rgb3:
                c_nb = _neighbor(col, dy, dx)
                best_c = np.where(sel[..., None], c_nb, best_c)
        frontier = (~filled) & any_nb
        if not frontier.any():
            break
        dep[frontier] = best_d[frontier]
        if rgb3:
            col[frontier] = best_c[frontier]
        filled |= frontier

    if rgb.dtype == np.uint8:
        col = np.clip(col, 0, 255).astype(np.uint8)
    return col, dep, filled
