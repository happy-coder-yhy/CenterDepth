"""纯 PyTorch 的 Softmax Splatting（前向投影），CPU / GPU 通用。

S²M² 官方推荐的 sniklaus/softmax-splatting 是 CUDA 算子；本工程在 Mac
（Apple Silicon，无 CUDA）上做可行性验证，因此用 scatter 实现等价的前向
软投影：每个源像素以 exp(weight) 加权投到目标像素，再做归一化。

面向中心视角合成的低成本融合改进（全部可开关）：
- 双向视差：右参考视差由右图直接推理，替代"从左图近似推导右流"；
- 光度校正：融合前把右图每通道 gain/offset 对齐到左图（在 pipeline 层）；
- 边缘感知权重：深度边缘（|∇d| 大）的源像素降权，减少跨边界混色重影；
- 视差中值滤波：消除视差斑点噪声；
- 权重语义（weight_mode）：修复"低置信/遮挡像素权重下限为 1"的缺陷，
  exp=旧行为（exp(conf·occ)∈[1,e]，不抑制）；linear=线性抑制
  （conf·occ∈[0,1]）；expdecay=指数衰减（exp(k(conf·occ−1))，k 控制抑制强度）；
- 深度一致性门控（可选）：两视图深度不一致时只保留更近的一侧；
- softz（软 z-buffer）：RGB 用深度+颜色一致性共同驱动的平滑选近，
  Depth 用门控选近，消除软平均的"半透明双影"；
- 深度 hard z-buffer（depth_z）：中心深度用"最近点胜出"而非软平均投影，
  恢复深度边缘锐度（软平均会把物体边缘抹糊）；
- 深度软 z 权重（depth_z_power）：左右两视图融合时按 1/depth^p 加权，
  更近的视图占主导，边缘锐利且无硬切换接缝；
- RGB 引导联合双边滤波（depth_jbf）：把中心深度边缘与 RGB 边缘对齐、
  收窄过渡带（物体边缘更锐利）；
- 背景深度遮挡填充：DIBR 孔洞填充，孔洞从"深度最大（背景）的邻居"逐层生长。
"""

from __future__ import annotations

import cv2
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


def hard_min_splatting(
    feature: torch.Tensor,
    flow: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """hard z-buffer 前向投影：每个目标像素取所有源中 feature 最小的源。

    与 softmax_splatting 的加权平均不同，这里不做平均、直接"最近胜出"，
    用于输出边缘锐利的中心深度（feature=深度时最小=最近）。参与竞争的源
    由 keep 掩码筛选（低置信/被遮挡的源被剔除）。

    Args:
        feature: (B, 1, H, W) 标量特征（深度）。
        flow: (B, 2, H, W) 前向流。
        keep: (B, 1, H, W) bool，参与 z-buffer 竞争的源像素。

    Returns:
        (B, 1, H, W) 各目标像素的最小 feature；无源投到处的像素为 0（无效）。
    """
    B, C, H, W = feature.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=feature.device),
        torch.arange(W, device=feature.device),
        indexing="ij",
    )
    tx = (xx.unsqueeze(0) + flow[:, 0]).round().long()
    ty = (yy.unsqueeze(0) + flow[:, 1]).round().long()
    inb = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H) & keep[:, 0]
    idx = (ty * W + tx).reshape(B, -1)
    v = inb.reshape(B, -1)
    out = torch.full(
        (B, H * W), float("inf"), dtype=feature.dtype, device=feature.device
    )
    for b in range(B):
        sel = idx[b][v[b]]
        f = feature[b, 0].reshape(-1)[v[b]]
        out[b].index_reduce_(0, sel, f, reduce="amin", include_self=False)
    out = out.reshape(B, 1, H, W)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _reliability_weight(
    conf: torch.Tensor,
    occ: torch.Tensor,
    edge: torch.Tensor | None,
    mode: str,
    k: float,
) -> torch.Tensor:
    """把 (conf, occ, edge) 转为传给 softmax_splatting 的 weight。

    softmax_splatting 内部取 exp(weight)，因此这里返回的 weight 需满足
    exp(weight) = 期望的"源像素可靠性"。基础可靠性 base = conf·occ·edge ∈ [0,1]。

    mode:
      - exp（旧行为）：weight = base → exp(base) ∈ [1, e]，低置信不抑制；
      - linear：weight = log(base) → exp = base ∈ [0,1]，线性抑制；
      - expdecay：weight = k·(base−1) → exp = exp(k(base−1)) ∈ [e^{−k}, 1]，
        k 越大低置信抑制越强（默认 k=4，权重下限 ≈ 0.018）。
    """
    base = (conf * occ).clamp(0.0, 1.0)
    if edge is not None:
        base = base * edge
    eps = 1e-6
    if mode == "exp":
        return base
    if mode == "linear":
        return torch.log(base + eps)
    if mode == "expdecay":
        return k * (base - 1.0)
    raise ValueError(f"未知权重语义: {mode}（可选 exp/linear/expdecay）")


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
    color_tol: float = 15.0,
    weight_mode: str = "exp",
    weight_k: float = 4.0,
    depth_z: bool = True,
    depth_z_thresh: float = 0.05,
    depth_z_power: float = 2.0,
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
        blend: softavg=置信度加权软平均；gate=深度一致性门控（不一致时只保留
               更近一侧）；hybrid=RGB 用软平均、Depth 用门控选近；conflict=
               软平均 + 冲突抑制（两视图深度或颜色差异大时平滑切换为单视图，
               消除近距离双影）；softz=软 z-buffer（RGB 用深度+颜色一致性
               平滑选近、Depth 门控选近，比 conflict 更平滑）。
        depth_tol: blend=gate 时的相对深度容差。
        color_tol: blend=softz/conflict 时的颜色冲突阈值（两 warp 平均色差，
          0-255；15 实测三帧重影归零，25 保留轻度残影）。
        weight_mode: 软投影权重语义（exp/linear/expdecay），见 _reliability_weight。
        weight_k: weight_mode=expdecay 时的抑制强度系数。
        depth_z: 中心深度用 hard z-buffer（最近点胜出）替代软平均，边缘更锐利。
        depth_z_thresh: 参与 z-buffer 的源可靠性阈值（基于 conf·occ，
          不含边缘降权；低于阈值的源不参与竞争）。
        depth_z_power: 跨视图深度融合的软 z 权重指数（w ∝ 1/depth^p，
          p 越大更近视图越占主导、边缘越锐利；0=普通加权平均）。

    Returns:
        center_rgb: (1, 3, H, W) 中心 RGB；
        center_depth: (1, 1, H, W) 中心深度（米），无效处为 0；
        valid: (1, 1, H, W) 有效掩码。
    """
    d_l = _median_filter(disp_left.clamp(min=0.0), median_k)
    edge_l = _edge_weight(d_l, edge_k) if edge_k > 0 else None
    weight_left = _reliability_weight(conf_left, occ_left, edge_l, weight_mode, weight_k)
    # 深度 z-buffer 参与掩码：基于 conf·occ（不含边缘降权），模式无关；
    # 无可靠源的像素回退软平均深度（见下），保证不产生空洞
    rel_left = (conf_left * occ_left).clamp(0.0, 1.0)
    zero = torch.zeros_like(d_l)

    # ---- 左视图 -> 中心 ----
    flow_l = torch.cat([-d_l / 2.0, zero], dim=1)
    depth_left = fx * baseline / d_l.clamp_min(0.5)
    rgb_l, norm_l, valid_l = softmax_splatting(left_rgb, flow_l, weight_left)
    if depth_z:
        dep_soft_l, _, _ = softmax_splatting(depth_left, flow_l, weight_left)
        dep_hard_l = hard_min_splatting(depth_left, flow_l, rel_left > depth_z_thresh)
        dep_l = torch.where(dep_hard_l > 0, dep_hard_l, dep_soft_l)
    else:
        dep_l, _, _ = softmax_splatting(depth_left, flow_l, weight_left)
    dep_ok_l = torch.isfinite(dep_l) & (dep_l > 0)

    # ---- 右视图 -> 中心 ----
    if disp_right is not None:
        d_r = _median_filter(disp_right.clamp(min=0.0), median_k)
        edge_r = _edge_weight(d_r, edge_k) if edge_k > 0 else None
        weight_right = _reliability_weight(conf_right, occ_right, edge_r, weight_mode, weight_k)
        rel_right = (conf_right * occ_right).clamp(0.0, 1.0)
        flow_r = torch.cat([d_r / 2.0, zero], dim=1)
        depth_right = fx * baseline / d_r.clamp_min(0.5)
        rgb_r, norm_r, valid_r = softmax_splatting(
            right_rgb, flow_r, weight_right
        )
        if depth_z:
            dep_soft_r, _, _ = softmax_splatting(depth_right, flow_r, weight_right)
            dep_hard_r = hard_min_splatting(depth_right, flow_r, rel_right > depth_z_thresh)
            dep_r = torch.where(dep_hard_r > 0, dep_hard_r, dep_soft_r)
        else:
            dep_r, _, _ = softmax_splatting(depth_right, flow_r, weight_right)
        dep_ok_r = torch.isfinite(dep_r) & (dep_r > 0)
    else:
        # 回退（旧行为）：把 d/2 场从左图坐标前向投影到右图坐标，得到 dR(xR)/2
        flow_l2r = torch.cat([-d_l, zero], dim=1)
        d_half_at_r, _, _ = softmax_splatting(d_l / 2.0, flow_l2r, weight_left)
        flow_r = torch.cat([d_half_at_r, zero], dim=1)
        depth_right = fx * baseline / (2.0 * d_half_at_r).clamp_min(0.5)
        ones = torch.ones_like(weight_left)
        rgb_r, norm_r, valid_r = softmax_splatting(right_rgb, flow_r, ones)
        dep_r, _, _ = softmax_splatting(depth_right, flow_r, ones)
        dep_ok_r = torch.isfinite(dep_r) & (dep_r > 0)

    # ---- 融合 ----
    if blend in ("gate", "hybrid", "conflict", "softz"):
        # 注意：softmax_splatting 返回的 valid 是"源像素是否投影出界"的掩码
        # （按源坐标索引），不是"目标像素是否收到贡献"。覆盖判断必须用
        # 目标像素上累计的权重和 norm（norm>0 表示至少一个源像素投到这里）。
        both = (norm_l > 1e-6) & (norm_r > 1e-6)
        agree = (dep_l - dep_r).abs() <= depth_tol * torch.minimum(dep_l, dep_r).clamp_min(0.1)
        closer_l = dep_l <= dep_r
        keep_l = agree | ~both | closer_l
        keep_r = agree | ~both | (~closer_l)
        if blend == "gate":
            w_l_rgb = w_l_dep = norm_l * keep_l.to(norm_l.dtype)
            w_r_rgb = w_r_dep = norm_r * keep_r.to(norm_r.dtype)
        elif blend in ("hybrid", "softz"):  # RGB 软平均/软 z-buffer，Depth 门控选近
            w_l_rgb, w_r_rgb = norm_l, norm_r
            w_l_dep = norm_l * keep_l.to(norm_l.dtype)
            w_r_dep = norm_r * keep_r.to(norm_r.dtype)
            if blend == "softz":
                # RGB 软 z-buffer：深度/颜色任一冲突时，平滑抑制更远一侧，
                # 消除软平均的"半透明双影"（比 conflict 的切换更平滑）
                dconf = (
                    (dep_l - dep_r).abs()
                    / (depth_tol * torch.minimum(dep_l, dep_r).clamp_min(0.1))
                    - 1.0
                ).clamp(0.0, 1.0)
                cdiff = (rgb_l - rgb_r).abs().mean(dim=1, keepdim=True)
                cconf = ((cdiff - color_tol) / max(color_tol, 1e-3)).clamp(0.0, 1.0)
                conflict = torch.maximum(dconf, cconf)
                # smoothstep：让"软平均 → 单视图"的切换没有可见折线
                s = conflict * conflict * (3.0 - 2.0 * conflict)
                both_f = both.to(norm_l.dtype)
                suppress = s * both_f  # 只有两视图都覆盖才抑制
                prefer_l = closer_l
                w_l_rgb = norm_l * (1.0 - suppress * (~prefer_l).to(norm_l.dtype))
                w_r_rgb = norm_r * (1.0 - suppress * prefer_l.to(norm_r.dtype))
        else:  # conflict：深度/颜色冲突处平滑切换为更近的单视图
            dconf = (
                (dep_l - dep_r).abs()
                / (depth_tol * torch.minimum(dep_l, dep_r).clamp_min(0.1))
                - 1.0
            ).clamp(0.0, 1.0)
            cdiff = (rgb_l - rgb_r).abs().mean(dim=1, keepdim=True)
            cconf = ((cdiff - color_tol) / max(color_tol, 1e-3)).clamp(0.0, 1.0)
            conflict = torch.maximum(dconf, cconf)
            prefer_l = closer_l
            w_l = norm_l * (1.0 - conflict * (~prefer_l).to(norm_l.dtype))
            w_r = norm_r * (1.0 - conflict * prefer_l.to(norm_r.dtype))
            # 只有两视图都覆盖时才做冲突抑制，单视图区域保持原贡献
            w_l = torch.where(both, w_l, norm_l)
            w_r = torch.where(both, w_r, norm_r)
            w_l_rgb = w_l_dep = w_l
            w_r_rgb = w_r_dep = w_r
    else:  # softavg
        w_l_rgb = w_l_dep = norm_l
        w_r_rgb = w_r_dep = norm_r
    # 深度贡献仅来自"有有效深度"的视图（hard z-buffer 空洞处不参与融合）
    w_l_dep = w_l_dep * dep_ok_l.to(w_l_dep.dtype)
    w_r_dep = w_r_dep * dep_ok_r.to(w_r_dep.dtype)
    if depth_z and depth_z_power > 0:
        # 软 z-buffer 跨视图权重：更近的视图占主导（比"agree 内平均"更锐利）
        w_l_dep = w_l_dep / dep_l.clamp_min(1e-3).pow(depth_z_power)
        w_r_dep = w_r_dep / dep_r.clamp_min(1e-3).pow(depth_z_power)
    wsum_rgb = w_l_rgb + w_r_rgb
    wsum_dep = w_l_dep + w_r_dep
    valid = (wsum_rgb > 1e-6) & (wsum_dep > 1e-6)
    center_rgb = (w_l_rgb * rgb_l + w_r_rgb * rgb_r) / wsum_rgb.clamp_min(1e-6)
    center_depth = (w_l_dep * dep_l + w_r_dep * dep_r) / wsum_dep.clamp_min(1e-6)
    center_depth = torch.where(
        torch.isfinite(center_depth), center_depth, torch.zeros_like(center_depth)
    )
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
    depth_tol: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于背景深度的 DIBR 遮挡填充（低成本孔洞修复）。

    中心视角中既无左图也无右图覆盖的孔洞（遮挡区）通常属于背景；
    这里让孔洞像素逐层从"深度最大（背景）的已填充邻居"生长，
    避免前景颜色/深度混入遮挡区。

    边界感知：8 邻域生长，且一旦孔洞有了背景深度估计，就拒绝与估计
    相差超过 depth_tol（相对值）的邻居——即不跨深度边缘取前景颜色，
    防止前景颜色沿孔洞"渗"进遮挡区。

    Args:
        rgb: (H, W, 3) float32/uint8 中心 RGB。
        depth: (H, W) float32 中心深度（米），孔洞处为 0。
        valid: (H, W) bool 有效掩码。
        max_iter: 最大生长迭代数（约等于最大孔洞宽度像素数）。
        depth_tol: 边界感知深度容差（相对值，0.3=允许邻居深度 ≥ 背景估计的 70%）。

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
        for dy, dx in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            d_nb = _neighbor(dep, dy, dx)
            v_nb = _neighbor(filled.astype(np.float32), dy, dx) > 0.5
            any_nb |= v_nb
            # 边界感知：已有背景估计的孔洞，拒绝与估计相差过大的邻居（前景）
            consistent = (best_d <= -np.inf) | (d_nb >= best_d * (1.0 - depth_tol))
            sel = v_nb & consistent & (d_nb >= best_d)
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


def joint_bilateral_depth(
    depth: np.ndarray,
    guide_rgb: np.ndarray,
    radius: int = 2,
    sigma_s: float = 2.0,
    sigma_c: float = 18.0,
    iters: int = 1,
) -> np.ndarray:
    """RGB 引导的联合双边滤波（中心深度边缘锐化）。

    深度值在颜色相近的区域被平滑，在颜色边缘处被保留——结果是把
    原本被软平均抹宽的深度过渡带收窄、并与 RGB 边缘对齐。

    Args:
        depth: (H, W) float32 中心深度（米），孔洞应已填充。
        guide_rgb: (H, W, 3) uint8 BGR 中心 RGB。
        radius: 空间窗口半径（像素）。
        sigma_s: 空间高斯 sigma（像素）。
        sigma_c: 颜色高斯 sigma（0-255 尺度）。
        iters: 迭代次数（1 足够；2 更强锐化）。

    Returns:
        锐化后的 (H, W) float32 深度。
    """
    d = np.asarray(depth, dtype=np.float32)
    g = cv2.cvtColor(np.asarray(guide_rgb), cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = d.shape
    cst = 2.0 * sigma_s * sigma_s
    ccc = 2.0 * sigma_c * sigma_c
    out = d.copy()
    for _ in range(iters):
        num = np.zeros((H, W), dtype=np.float32)
        den = np.zeros((H, W), dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                w = float(np.exp(-(dy * dy + dx * dx) / cst))
                dq = _neighbor(d, dy, dx)
                gq = _neighbor(g, dy, dx)
                cd = ((g - gq) ** 2).sum(axis=2)
                wq = w * np.exp(-cd / ccc)
                num += wq * dq
                den += wq
        out = num / np.maximum(den, 1e-6)
        d = out
    return out
