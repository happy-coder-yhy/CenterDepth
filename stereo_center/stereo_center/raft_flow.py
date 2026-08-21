"""RAFT 光流轻量封装（直接使用 StereoAnyVideo 仓库 vendored 的 RAFT 代码）。

官方 RAFTModel 包装依赖 pytorch3d，这里绕过它直接加载 core/raft.py，
仅需 torch + 权重文件 raft-things.pth。

约定：
    load_raft(weights_path, raft_root, device) -> model
    flow_between(model, img0, img1, iters) -> (1, 2, H, W) float32 flow
img0/img1: (1, 3, H, W) 0-255 float RGB（RAFT 内部自行归一化）；
flow 方向：img0 坐标系的像素位移，指向 img1（warp 时目标坐标 = 源坐标 + flow）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _resolve_raft_root(explicit: str | Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    candidates += [
        Path.home() / "BothEyesDepth" / "stereoanyvideo" / "third_party" / "RAFT",
        Path(__file__).resolve().parents[2] / "third_party" / "RAFT",
    ]
    for c in candidates:
        if (c / "core" / "raft.py").exists():
            c = c.resolve()
            # 用 stereoanyvideo 仓库根的命名空间路径导入，避免与其他仓库的 core 包撞名
            sav_root = c.parent.parent
            for p in (str(sav_root.parent), str(sav_root), str(c)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return c
    raise FileNotFoundError(
        "未找到 RAFT 仓库代码，请用 --raft-root 或 RAFT_ROOT 指定"
    )


def load_raft(
    weights_path: str | Path,
    raft_root: str | Path | None = None,
    device: str = "cuda",
    small: bool = False,
) -> torch.nn.Module:
    _resolve_raft_root(raft_root)
    from stereoanyvideo.third_party.RAFT.core.raft import RAFT

    args = SimpleNamespace(mixed_precision=False, small=small, dropout=0.0)
    model = RAFT(args)
    state = torch.load(Path(weights_path), map_location="cpu")
    state = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"[raft] RAFT 光流权重加载完成: {weights_path} (device={device})")
    return model


@torch.no_grad()
def flow_between(
    model: torch.nn.Module,
    img0: torch.Tensor,
    img1: torch.Tensor,
    iters: int = 20,
) -> torch.Tensor:
    """计算 img0 -> img1 的全分辨率光流。"""
    from stereoanyvideo.third_party.RAFT.core.utils.utils import InputPadder

    padder = InputPadder(img0.shape)
    a, b = padder.pad(img0, img1)
    _flow_low, flow_up = model(a, b, iters=iters, test_mode=True)
    return padder.unpad(flow_up)
