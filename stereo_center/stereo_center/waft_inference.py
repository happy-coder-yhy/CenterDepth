"""WAFT-Stereo 模型加载与推理封装（GPU 推理；CPU 仅用于形状冒烟）。

与 s2m2_inference 保持一致的调用约定：
    load_waft(model_type, weights_dir, device) -> model
    run_stereo_matching(model, left, right, device, **kwargs)
        -> (disp, occ, conf, elapsed)

输入张量：left/right 为 (1, 3, H, W) 0-255 float RGB（与 S²M² 封装一致）；
输出：disp/occ/conf 均为 (H, W) float32 CPU 张量。

WAFT-Stereo 原生只输出视差与 4 通道 Laplacian 混合参数（info），
没有 occlusion/confidence，因此：
- occ 由可见性（x-d>=0）或左右一致性（默认）计算；
- conf 默认取官方 uncertainty 映射 softmax(info[:, :2])[:, 0]，也可用常量 1。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # stereo_center/
WAFT_ROOT = PROJECT_ROOT / "third_party" / "waft"
if str(WAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(WAFT_ROOT))

from peft import PeftModel  # noqa: E402
from bridgedepth.config import get_cfg  # noqa: E402
from algorithms.waft import WAFT  # noqa: E402

WAFT_MODEL_CONFIG = {
    "DAv2S-4": "configs/SynLarge/DAv2S-4.yaml",
    "DAv2B-4": "configs/SynLarge/DAv2B-4.yaml",
    "DAv2L-5": "configs/SynLarge/DAv2L-5.yaml",
}


def load_waft(
    model_type: str = "DAv2L-5",
    weights_dir: str | Path = "weights/waft",
    device: str = "cuda",
) -> WAFT:
    """加载 WAFT-Stereo 预训练权重（ckpt 内含完整权重，无需另下 DAv2）。"""
    if model_type not in WAFT_MODEL_CONFIG:
        raise ValueError(
            f"未知 WAFT 模型类型: {model_type}（可选: {', '.join(WAFT_MODEL_CONFIG)}）"
        )
    ckpt_path = Path(weights_dir) / f"{model_type}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"权重文件不存在: {ckpt_path}\n"
            f"请确认 --weights / WAFT_WEIGHTS_DIR 指向包含该文件的目录，"
            f"或运行 scripts/download_waft_weights.sh 下载。"
        )

    cfg = get_cfg()
    cfg.merge_from_file(str(WAFT_ROOT / WAFT_MODEL_CONFIG[model_type]))
    cfg.freeze()
    model = WAFT(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(weights, strict=False)
    # 与官方 demo 一致：加载后把 LoRA 适配器合并回骨干权重
    for _name, module in model.named_modules():
        if isinstance(module, PeftModel):
            module.merge_and_unload()

    model.eval().to(device)
    print(f"[waft] {model_type} 权重加载完成: {ckpt_path}")
    return model


def _run_once(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    hiera_mode: str,
    use_amp: bool,
) -> tuple[dict, float]:
    """单方向推理：direct 用 model(sample)，hiera 用 0.5->1.0 由粗到细。"""
    sample = {"img1": left, "img2": right}
    if hiera_mode == "hiera":
        def forward():
            return model.heirarchical_inference(
                sample, size=None, factor_list=[0.5, 1.0]
            )
    else:
        forward = lambda: model(sample)

    t0 = time.time()
    if use_amp and left.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = forward()
    else:
        out = forward()
    elapsed = time.time() - t0
    return out, elapsed


def _visibility_mask(disp: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """左图像素的右图对应点仍在图内：x - d >= 0。"""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=disp.device),
        torch.arange(W, device=disp.device),
        indexing="ij",
    )
    return (xx - disp[0] >= 0).float()


def _lr_consistency_mask(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """左右一致性：|dL - dR(x-dL)| < max(1.0, 5% * dL) 且 x-dL >= 0。"""
    dL = disp_l[0]
    dR = disp_r[0]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=dL.device),
        torch.arange(W, device=dL.device),
        indexing="ij",
    )
    tx = xx - dL
    grid = torch.stack(
        [2.0 * tx / max(W - 1, 1) - 1.0, 2.0 * yy / max(H - 1, 1) - 1.0],
        dim=-1,
    ).unsqueeze(0)
    dR_at_l = F.grid_sample(
        dR.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    diff = (dL - dR_at_l).abs()
    thresh = torch.maximum(torch.full_like(dL, 1.0), 0.05 * dL)
    return ((diff < thresh) & (xx - dL >= 0)).float()


def _resolve_hiera(hiera: str, H: int, W: int) -> str:
    """auto：max(H,W)>1080 用 0.5->1.0 分层，否则直接 forward。"""
    if hiera == "auto":
        return "hiera" if max(H, W) > 1080 else "direct"
    return hiera


def _conf_from_info(info: torch.Tensor, conf_mode: str, disp: torch.Tensor) -> torch.Tensor:
    """info：官方 uncertainty 映射 softmax(info[:, :2])[:, 0]；ones：常量 1。"""
    if conf_mode == "ones":
        return torch.ones_like(disp[0])
    weight = info[:, :2].softmax(dim=1)
    return weight[0, 0]


@torch.no_grad()
def run_stereo_matching(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    hiera: str = "auto",
    conf_mode: str = "info",
    occ_mode: str = "lr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """推理 WAFT-Stereo。

    Args:
        left/right: (1, 3, H, W) 0-255 float RGB tensor。
        hiera: auto（max(H,W)>1080 用 0.5->1.0 分层）/ direct / hiera。
        conf_mode: info（官方 uncertainty 映射）/ ones。
        occ_mode: lr（左右一致性，双向推理）/ visibility（仅 x-d>=0，单向前向）。

    Returns:
        disp/occ/conf: (H, W) float32 CPU tensor；
        elapsed: 推理耗时（秒，lr 模式含双向前向）。
    """
    H, W = left.shape[-2:]
    hiera_mode = _resolve_hiera(hiera, H, W)
    if hiera_mode not in ("direct", "hiera"):
        raise ValueError(f"未知推理模式: {hiera}（可选 auto/direct/hiera）")
    if conf_mode not in ("info", "ones"):
        raise ValueError(f"未知置信度模式: {conf_mode}（可选 info/ones）")
    if occ_mode not in ("lr", "visibility"):
        raise ValueError(f"未知遮挡模式: {occ_mode}（可选 lr/visibility）")

    lt = left.to(device)
    rt = right.to(device)
    out_l, t1 = _run_once(model, lt, rt, hiera_mode, use_amp)
    disp = out_l["disp_pred"]  # (B, H, W)
    info = out_l["delta_info_preds"][-1]  # (B, 4, H, W)

    if occ_mode == "lr":
        out_r, t2 = _run_once(model, rt, lt, hiera_mode, use_amp)
        occ = _lr_consistency_mask(disp, out_r["disp_pred"], H, W)
        elapsed = t1 + t2
    else:
        occ = _visibility_mask(disp, H, W)
        elapsed = t1

    conf = _conf_from_info(info, conf_mode, disp)

    return disp[0].float().cpu(), occ.float().cpu(), conf.float().cpu(), elapsed


@torch.no_grad()
def run_stereo_matching_bi(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    hiera: str = "auto",
    conf_mode: str = "info",
    occ_mode: str = "lr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """双向视差推理（左右参考各一次前向）。

    与 run_stereo_matching(occ_mode="lr") 的两次前向等价，但额外返回
    右参考视差/遮挡/置信度，供中心视角融合直接使用真实 dR。

    Returns:
        (dL, dR, occL, occR, confL, confR, elapsed)，均为 (H, W) float32 CPU。
    """
    H, W = left.shape[-2:]
    hiera_mode = _resolve_hiera(hiera, H, W)
    if hiera_mode not in ("direct", "hiera"):
        raise ValueError(f"未知推理模式: {hiera}（可选 auto/direct/hiera）")
    if conf_mode not in ("info", "ones"):
        raise ValueError(f"未知置信度模式: {conf_mode}（可选 info/ones）")
    if occ_mode not in ("lr", "visibility"):
        raise ValueError(f"未知遮挡模式: {occ_mode}（可选 lr/visibility）")

    lt = left.to(device)
    rt = right.to(device)
    out_l, t1 = _run_once(model, lt, rt, hiera_mode, use_amp)
    out_r, t2 = _run_once(model, rt, lt, hiera_mode, use_amp)
    dL = out_l["disp_pred"]
    dR = out_r["disp_pred"]
    if occ_mode == "visibility":
        occL = _visibility_mask(dL, H, W)
        occR = _visibility_mask(dR, H, W)
    else:
        occL = _lr_consistency_mask(dL, dR, H, W)
        occR = _lr_consistency_mask(dR, dL, H, W)
    confL = _conf_from_info(out_l["delta_info_preds"][-1], conf_mode, dL)
    confR = _conf_from_info(out_r["delta_info_preds"][-1], conf_mode, dR)
    return (
        dL[0].float().cpu(),
        dR[0].float().cpu(),
        occL.float().cpu(),
        occR.float().cpu(),
        confL.float().cpu(),
        confR.float().cpu(),
        t1 + t2,
    )
