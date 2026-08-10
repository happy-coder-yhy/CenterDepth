"""纯 PyTorch 的 Softmax Splatting（前向投影），CPU / GPU 通用。

S²M² 官方推荐的 sniklaus/softmax-splatting 是 CUDA 算子；本工程在 Mac
（Apple Silicon，无 CUDA）上做可行性验证，因此用 scatter 实现等价的前向
软投影：每个源像素以 exp(weight) 加权投到目标像素，再做归一化。
"""

from __future__ import annotations

import torch


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """由校正后双目 + S²M² 结果合成中心视角 RGB 与深度。

    几何：中心像素 xM = (xL + xR) / 2
      - 左图到中心 flow: u_L = -d/2
      - 右图到中心 flow: u_R = +dR/2，其中右视差 dR(xR) = dL(xL)，
        通过把 d/2 场从左图前向投影到右图坐标得到（一步近似）。

    Args:
        left_rgb/right_rgb: (1, 3, H, W) 0-255 float tensor（校正后）。
        disp_left: (1, 1, H, W) 左视差 d = xL - xR。
        conf_left/occ_left: (1, 1, H, W) S²M² 置信度 / 遮挡掩码。
        fx: 校正后焦距（像素）；baseline: 基线（米）。

    Returns:
        center_rgb: (1, 3, H, W) 中心 RGB；
        center_depth: (1, 1, H, W) 中心深度（米），无效处为 0；
        valid: (1, 1, H, W) 有效掩码。
    """
    disp = disp_left.clamp(min=0.0)
    weight_left = (conf_left * occ_left).clamp(0.0, 1.0)
    zero = torch.zeros_like(disp)

    # ---- 左视图 -> 中心 ----
    flow_l = torch.cat([-disp / 2.0, zero], dim=1)
    depth_left = fx * baseline / disp.clamp_min(0.5)
    rgb_l, norm_l, _ = softmax_splatting(left_rgb, flow_l, weight_left)
    dep_l, _, _ = softmax_splatting(depth_left, flow_l, weight_left)

    # ---- 右视图 -> 中心 ----
    # 把 d/2 场从左图坐标前向投影到右图坐标，得到 dR(xR)/2
    flow_l2r = torch.cat([-disp, zero], dim=1)
    d_half_at_r, _, _ = softmax_splatting(disp / 2.0, flow_l2r, weight_left)
    flow_r = torch.cat([d_half_at_r, zero], dim=1)
    depth_right = fx * baseline / (2.0 * d_half_at_r).clamp_min(0.5)
    rgb_r, norm_r, _ = softmax_splatting(right_rgb, flow_r, torch.ones_like(weight_left))
    dep_r, _, _ = softmax_splatting(depth_right, flow_r, torch.ones_like(weight_left))

    # ---- 置信度加权融合 ----
    wsum = norm_l + norm_r
    valid = wsum > 1e-6
    center_rgb = (norm_l * rgb_l + norm_r * rgb_r) / wsum.clamp_min(1e-6)
    center_depth = (norm_l * dep_l + norm_r * dep_r) / wsum.clamp_min(1e-6)
    center_depth = center_depth * valid.to(center_depth.dtype)
    return center_rgb, center_depth, valid
