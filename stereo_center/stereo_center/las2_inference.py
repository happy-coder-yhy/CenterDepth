"""Lite Any Stereo V2 (LAS2) 推理封装（逐帧前向，与 WAFT/S²M² 接口对齐）。

LAS2 是轻量前馈立体匹配模型（FasterNet-T0 特征 + 相关体 + FasterNet 聚合 +
上下文上采样），单次前向即出全分辨率左参考视差，无迭代循环，非常适合
逐帧批量推理（视频管线直接复用 run_depth_video.py）。

调用约定（与 waft_inference / s2m2_inference 一致）：
    load_las2(model_size, weights_dir, device, max_disp) -> model
    run_stereo_matching(model, left, right, device, **kw)
        -> (disp, occ, conf, elapsed)，均为 (H, W) float32 CPU
    run_stereo_matching_bi_batch(model, left, right, device, **kw)
        -> (dL, dR, occL, occR, confL, confR, elapsed)，均为 (B, H, W) float32 CPU

输入张量：left/right 为 (B, 3, H, W) 0-255 float RGB；输出为正的左/右参考
视差（像素）。LAS2 无原生遮挡/置信度输出：occ 默认可见性（x-d>=0），
可选左右一致性；conf 用常量 1。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # stereo_center/


def _resolve_las_root(explicit: str | Path | None = None) -> Path:
    """定位 LiteAnyStereo 仓库根目录并加入 sys.path。"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if "LAS_ROOT" in os.environ:
        candidates.append(Path(os.environ["LAS_ROOT"]).resolve())
    candidates += [
        PROJECT_ROOT.parent / "LiteAnyStereo",
        Path.home() / "BothEyesDepth" / "LiteAnyStereo",
    ]
    for c in candidates:
        if (c / "core").exists():
            c = c.resolve()
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise FileNotFoundError(
        "未找到 LiteAnyStereo 仓库，请用 --las-root 或环境变量 LAS_ROOT 指定"
    )


def load_las2(
    model_size: str = "M",
    weights_dir: str | Path = "weights/pretrain_weights",
    device: str = "cuda",
    max_disp: int = 192,
    las_root: str | Path | None = None,
) -> torch.nn.Module:
    """加载 LAS2 预训练权重（LAS2_S/M/L/H.pth，HuggingFace 官方发布）。"""
    _resolve_las_root(las_root)
    from core.models import build_model, load_model_weights, normalize_las2_model_size

    size = normalize_las2_model_size(model_size)
    ckpt_path = Path(weights_dir) / f"LAS2_{size.upper()}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"LAS2 权重不存在: {ckpt_path}\n"
            f"请运行 LiteAnyStereo/download_checkpoint.py 下载后放到该目录，"
            f"或用 --weights / LAS2_WEIGHTS_DIR 指定。"
        )
    model = build_model("las2", fnet_pretrained=False, model_size=size, max_disp=max_disp)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_model_weights(model, ckpt, strict=True)
    model.eval().to(device)
    print(f"[las2] LAS2-{size.upper()} 权重加载完成: {ckpt_path} "
          f"(device={device}, max_disp={max_disp})")
    return model


def _run_once(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    max_disp: int,
    use_amp: bool,
) -> tuple[torch.Tensor, float]:
    """单向前向，返回 (B, H, W) float32 CPU 参考视差。"""
    from core.utils.utils import InputPadder

    padder = InputPadder(left.shape, divis_by=32)
    l, r = padder.pad(left, right)
    t0 = time.time()
    with torch.no_grad():
        if use_amp and left.is_cuda:
            with torch.autocast("cuda", enabled=True):
                disp = model(l, r, max_disp=max_disp, test_mode=True)
        else:
            disp = model(l, r, max_disp=max_disp, test_mode=True)
    t = time.time() - t0
    return padder.unpad(disp.float())[:, 0].cpu(), t


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    B, H, W = disp.shape
    x = torch.arange(W, device=disp.device).view(1, 1, W)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < W - 1)).float()


def _lr_consistency(dL: torch.Tensor, dR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """左右一致性遮挡：|dL(x) - dR(x - dL(x))| < max(1, 0.05*dL)。"""
    B, H, W = dL.shape
    yy = torch.linspace(-1, 1, H, device=dL.device).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=dL.device).float()
    grid_l = torch.stack(
        [((xx - dL) / max(W - 1, 1) * 2 - 1).clamp(-1, 1), yy], dim=-1
    )  # (B,H,W,2) 采样 dR 在左坐标
    dR_at_L = F.grid_sample(
        dR.unsqueeze(1), grid_l, mode="bilinear", align_corners=True, padding_mode="zeros"
    )[:, 0]
    ok_l = ((dL - dR_at_L).abs() <= torch.maximum(torch.tensor(1.0, device=dL.device), 0.05 * dL)) & (xx.view(1, 1, W) - dL >= 0)
    grid_r = torch.stack(
        [((xx - dR) / max(W - 1, 1) * 2 - 1).clamp(-1, 1), yy], dim=-1
    )
    dL_at_R = F.grid_sample(
        dL.unsqueeze(1), grid_r, mode="bilinear", align_corners=True, padding_mode="zeros"
    )[:, 0]
    ok_r = ((dR - dL_at_R).abs() <= torch.maximum(torch.tensor(1.0, device=dL.device), 0.05 * dR)) & (xx.view(1, 1, W) - dR >= 0)
    return ok_l.float(), ok_r.float()


def run_stereo_matching(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    max_disp: int = 192,
    use_amp: bool = True,
    conf_mode: str = "ones",
    occ_mode: str = "visibility",
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """单方向推理：返回 (disp, occ, conf, elapsed)，均为 (H, W) float32 CPU。"""
    d, t = _run_once(model, left.to(device), right.to(device), max_disp, use_amp)
    occ = _visibility(d)
    conf = torch.ones_like(d)
    if d.shape[0] == 1:
        return d[0], occ[0], conf[0], t
    return d, occ, conf, t


def run_stereo_matching_bi_batch(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    max_disp: int = 192,
    use_amp: bool = True,
    conf_mode: str = "ones",
    occ_mode: str = "visibility",
    batch: int = 8,
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """批量双向推理：交换输入得到右参考视差。

    Returns: (dL, dR, occL, occR, confL, confR, elapsed)，均为 (B, H, W) float32 CPU。
    """
    B = left.shape[0]
    batch = max(1, min(int(batch), B))
    dL_parts, dR_parts = [], []
    t_total = 0.0
    for s in range(0, B, batch):
        e = min(s + batch, B)
        lt = left[s:e].to(device)
        rt = right[s:e].to(device)
        dLc, t1 = _run_once(model, lt, rt, max_disp, use_amp)
        dRc, t2 = _run_once(model, rt, lt, max_disp, use_amp)
        t_total += t1 + t2
        dL_parts.append(dLc)
        dR_parts.append(dRc)
    dL = torch.cat(dL_parts, dim=0)
    dR = torch.cat(dR_parts, dim=0)
    if occ_mode == "lr":
        occL, occR = _lr_consistency(dL.to(device), dR.to(device))
        occL, occR = occL.cpu(), occR.cpu()
    else:
        occL = _visibility(dL)
        occR = _visibility(dR)
    confL = torch.ones_like(dL)
    confR = torch.ones_like(dR)
    return dL, dR, occL, occR, confL, confR, t_total


def run_stereo_matching_bi(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    **kwargs,
):
    """单帧双向（兼容旧接口）：输入 (1,3,H,W)，返回 (H,W) CPU。"""
    out = run_stereo_matching_bi_batch(model, left, right, device, **kwargs)
    return tuple(x[0] for x in out[:6]) + (out[6],)
