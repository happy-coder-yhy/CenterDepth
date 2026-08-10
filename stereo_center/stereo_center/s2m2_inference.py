"""S²M² 模型加载与推理封装（CPU / GPU 通用）。"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from s2m2.core.model.s2m2 import S2M2
from s2m2.core.utils.image_utils import image_crop, image_pad

MODEL_CONFIG = {
    "S": {"feature_channels": 128, "n_transformer": 1},
    "M": {"feature_channels": 192, "n_transformer": 2},
    "L": {"feature_channels": 256, "n_transformer": 3},
    "XL": {"feature_channels": 384, "n_transformer": 3},
}


def load_s2m2(
    model_type: str = "S",
    weights_dir: str | Path = "weights/pretrain_weights",
    refine_iter: int = 3,
    device: str = "cpu",
) -> S2M2:
    """加载 S²M² 预训练权重（自动处理 CUDA 存档到 CPU 的映射）。"""
    cfg = MODEL_CONFIG[model_type]
    model = S2M2(
        feature_channels=cfg["feature_channels"],
        dim_expansion=1,
        num_transformer=cfg["n_transformer"],
        use_positivity=True,
        refine_iter=refine_iter,
    )
    ckpt_path = Path(weights_dir) / f"CH{cfg['feature_channels']}NTR{cfg['n_transformer']}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"权重文件不存在: {ckpt_path}\n"
            f"请确认 --weights / S2M2_WEIGHTS_DIR 指向包含该文件的目录。"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.my_load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"[s2m2] {model_type} 权重加载完成: {ckpt_path}")
    return model


@torch.no_grad()
def run_stereo_matching(
    model: S2M2,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cpu",
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """推理 S²M²。

    Args:
        left/right: (1, 3, H, W) 0-255 float tensor。

    Returns:
        disp/occ/conf: (H, W) float32 CPU tensor；
        elapsed: 推理耗时（秒）。
    """
    H, W = left.shape[-2:]
    left_pad = image_pad(left, 32).to(device)
    right_pad = image_pad(right, 32).to(device)

    t0 = time.time()
    if use_amp and torch.device(device).type != "cpu":
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16):
            disp, occ, conf = model(left_pad, right_pad)
    else:
        # 默认 FP32：本机实测 FP32 在 A100 上更快且精度一致
        disp, occ, conf = model(left_pad, right_pad)
    elapsed = time.time() - t0

    disp = image_crop(disp, (H, W)).squeeze().float().cpu()
    occ = image_crop(occ, (H, W)).squeeze().float().cpu()
    conf = image_crop(conf, (H, W)).squeeze().float().cpu()
    return disp, occ, conf, elapsed
