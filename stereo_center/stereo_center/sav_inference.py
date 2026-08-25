"""StereoAnyVideo 视频级立体匹配封装（ICCV 2025，时间一致）。

与 waft_inference 的区别：StereoAnyVideo 以"整段视频"为输入，时间一致性来自
kernel=20 / stride=10 的滑窗 + 时间注意力，官方 demo 一次传入整段视频、
由 forward_batch_test 内部逐块前向。本封装：

    load_sav(sav_root, ckpt_path, device) -> SavStereo
    SavStereo.run_segment(left, right, iters, target_len, bidirectional)
        -> (dL, dR)（(target_len, H, W) float32 numpy，左/右参考视差，像素）

输入张量约定：left/right 为 (T, 3, H, W) 0-255 float RGB（CPU 即可，
forward_batch_test 内部按块搬到 cuda）；输出分辨率与输入一致
（内部 InputPadder 按 32 对齐后 unpad）。

注意：官方模型只输出单侧视差，无遮挡/置信度输出；occ 由可见性计算、
conf 用常量 1（在 run_depth_video_sav.py 中处理）。VDA（Video-Depth-Anything）
权重由 DepthExtractor 在构造时按 CWD 相对路径加载，因此构造前会临时
chdir 到 sav_root（上游未提供绝对路径参数，保持上游代码零修改）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _add_sav_to_path(sav_root: str | Path) -> Path:
    """把 stereoanyvideo 仓库根目录加入 sys.path（namespace package，无 __init__.py）。"""
    sav_root = Path(sav_root).resolve()
    # 包名 stereoanyvideo 即仓库根目录，需把其父目录放入 path
    for p in (str(sav_root.parent), str(sav_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return sav_root


def _load_state_dict(model: torch.nn.Module, ckpt_path: Path) -> None:
    """加载官方 ckpt，兼容 model / state_dict / module. 前缀。"""
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def _pad_to(video: torch.Tensor, target_len: int) -> torch.Tensor:
    """不足 target_len 帧时用末帧复制补齐（forward_batch_test 对尾部有截断）。"""
    t = video.shape[0]
    if t >= target_len:
        return video
    tail = video[-1:].expand(target_len - t, -1, -1, -1)
    return torch.cat([video, tail], dim=0)


class SavStereo:
    """StereoAnyVideo 推理封装。"""

    def __init__(self, model: torch.nn.Module, sav_root: Path, device: str):
        self.model = model
        self.sav_root = Path(sav_root)
        self.device = device

    @torch.no_grad()
    def _forward(
        self, left: torch.Tensor, right: torch.Tensor, iters: int
    ) -> np.ndarray:
        """前向一次，返回 (T, H, W) float32 左参考视差（绝对像素）。"""
        video = torch.stack([left, right], dim=1)  # (T, 2, 3, H, W)
        batch_dict = {"stereo_video": video}
        preds = self.model.forward_batch_test(batch_dict, iters=iters)
        disp = preds["disparity"][:, :1].clone().cpu().abs().numpy()[:, 0]
        return disp

    def run_segment(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        iters: int = 20,
        target_len: int = 400,
        bidirectional: bool = True,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """对一段视频推理，输出该段（补齐到 target_len 帧）的视差。

        Returns:
            dL: (target_len, H, W) float32 左参考视差；
            dR: 双向时 (target_len, H, W) 右参考视差，否则 None。
        """
        target_len = max(target_len, 30)
        left = _pad_to(left, target_len)
        right = _pad_to(right, target_len)

        t0 = time.time()
        dL = self._forward(left, right, iters)
        t_forward = time.time() - t0
        dR = None
        if bidirectional:
            t0 = time.time()
            dR = self._forward(right, left, iters)  # 交换输入即右参考视差
            t_forward += time.time() - t0
        if verbose:
            rate = target_len / max(t_forward, 1e-6)
            print(
                f"[sav] 段长 {target_len} 帧（双向={bidirectional}）耗时 "
                f"{t_forward:.1f}s，吞吐 {rate:.2f} 帧/s，视差范围 "
                f"[{dL.min():.1f}, {dL.max():.1f}]",
                flush=True,
            )
        return dL, dR


def load_sav(
    sav_root: str | Path,
    ckpt_path: str | Path,
    device: str = "cuda",
    amp: bool = False,
    tf32: bool = True,
    bench: bool = False,
) -> SavStereo:
    """构造 StereoAnyVideo 模型并加载官方 checkpoint。

    sav_root: 官方仓库根目录（含 models/、checkpoints/ 等）；
    ckpt_path: StereoAnyVideo_SF.pth / StereoAnyVideo_MIX.pth 路径。
    """
    sav_root = _add_sav_to_path(sav_root)
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"StereoAnyVideo 权重不存在: {ckpt_path}\n"
            f"请在服务器上运行 gdown --folder 1c7L065dcBWhCYYjWYo2edGOG605PnpXv "
            f"下载后放到 sav_root/checkpoints/ 下。"
        )

    # DepthExtractor 构造时按 CWD 相对路径加载 VDA 权重，临时切到 sav_root
    cwd = os.getcwd()
    os.chdir(sav_root)
    try:
        from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo

        model = StereoAnyVideo(mixed_precision=bool(amp))
    finally:
        os.chdir(cwd)

    _load_state_dict(model, ckpt_path)
    if amp or bench:
        # 固定输入尺寸下开启 cudnn benchmark（自动选择最优卷积算法）
        torch.backends.cudnn.benchmark = True
    if tf32 or amp:
        # A100 TF32：FP32 矩阵乘/卷积提速明显，精度损失可忽略（推理常用）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.eval().to(device)
    print(f"[sav] StereoAnyVideo 权重加载完成: {ckpt_path} "
          f"(device={device}, amp={bool(amp)}, tf32={bool(tf32)}, bench={bool(bench)})")
    return SavStereo(model, sav_root, device)
