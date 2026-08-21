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
from contextlib import contextmanager, nullcontext
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
from stereo_center.temporal_stereo import temporal_alignment_mask  # noqa: E402

WAFT_MODEL_CONFIG = {
    "DAv2S-4": "configs/SynLarge/DAv2S-4.yaml",
    "DAv2B-4": "configs/SynLarge/DAv2B-4.yaml",
    "DAv2L-5": "configs/SynLarge/DAv2L-5.yaml",
}


class _TimingRecorder:
    """WAFT wall-clock stage recorder; CUDA synchronization makes boundaries real."""

    def __init__(self, device: str):
        self.device = device
        self.stages: dict[str, float] = {}

    def _sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0


def load_waft(
    model_type: str = "DAv2L-5",
    weights_dir: str | Path = "weights/waft",
    device: str = "cuda",
    iters: int | None = None,
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
    if iters is not None:
        cfg.WAFT.ITERATIVE_MODULE.TASK = ["delta"] * int(iters)
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
    print(f"[waft] {model_type} 权重加载完成: {ckpt_path} (iters={len(cfg.WAFT.ITERATIVE_MODULE.TASK)})")
    return model


def _pack_temporal_reference_pairs(
    left: torch.Tensor,
    right: torch.Tensor,
    temporal_state: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack left and flipped-right frames as two independent time groups."""
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("left and right must have identical (B, C, H, W) shapes")
    batch = left.shape[0]
    current = torch.cat((left, torch.flip(right, dims=[3])), dim=0)
    previous = current.clone()
    boundary = torch.ones(
        current.shape[0], 1, current.shape[2], current.shape[3],
        device=current.device, dtype=current.dtype,
    )
    carry_images = temporal_state.get("previous_reference_images")
    for group, start in enumerate((0, batch)):
        end = start + batch
        if batch > 1:
            previous[start + 1:end] = current[start:end - 1]
        if carry_images is None:
            boundary[start] = 0.0
        else:
            if carry_images.shape != (2, *current.shape[1:]):
                raise ValueError("temporal carry images do not match the current frame shape")
            previous[start] = carry_images[group].to(current)
    return current, previous, boundary


def _run_once(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    hiera_mode: str,
    use_amp: bool,
    timing: _TimingRecorder | None = None,
    temporal_init: dict | None = None,
) -> tuple[dict, float]:
    """单方向推理：direct 用 model(sample)，hiera 用 0.5->1.0 由粗到细。"""
    sample = {"img1": left, "img2": right}
    if hiera_mode == "hiera":
        if temporal_init is not None:
            raise ValueError("WAFT temporal initialization supports direct inference only")
        def forward():
            return model.heirarchical_inference(
                sample, size=None, factor_list=[0.5, 1.0], timing=timing
            )
    else:
        forward = lambda: model(sample, temporal_init=temporal_init, timing=timing)

    if left.device.type == "cuda":
        torch.cuda.synchronize(left.device)
    t0 = time.perf_counter()
    if timing is not None:
        timing_context = timing.measure("model_forward")
    else:
        timing_context = nullcontext()
    with timing_context:
        if use_amp and left.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward()
        else:
            out = forward()
    if left.device.type == "cuda":
        torch.cuda.synchronize(left.device)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _run_bidirectional_once(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    hiera_mode: str,
    use_amp: bool,
    timing: _TimingRecorder | None = None,
    temporal_init: dict | None = None,
) -> tuple[dict, int, float]:
    """把左右两个方向拼为一个 2B 前向，减少一次模型调度和重复开销。"""
    batch = left.shape[0]
    if timing is not None:
        pack_context = timing.measure("bidirectional_pack")
    else:
        pack_context = nullcontext()
    with pack_context:
        left_bi = torch.cat((left, torch.flip(right, dims=[3])), dim=0)
        right_bi = torch.cat((right, torch.flip(left, dims=[3])), dim=0)
    out, elapsed = _run_once(
        model, left_bi, right_bi, hiera_mode, use_amp,
        timing=timing, temporal_init=temporal_init,
    )
    return out, batch, elapsed


def _visibility_mask(disp: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """左图像素的右图对应点仍在图内：x - d >= 0。"""
    yy, xx = torch.meshgrid(
        torch.arange(H, device=disp.device),
        torch.arange(W, device=disp.device),
        indexing="ij",
    )
    return (xx - disp[0] >= 0).float()


def _lr_error(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """左右一致性误差 |dL - dR(x-dL)| 与右图内坐标 x-dL。"""
    diff, tx = _lr_error_batch(disp_l, disp_r, H, W)
    return diff[0], tx[0]


def _lr_error_batch(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """批量计算左右一致性误差，避免逐帧调用 grid_sample。"""
    dL = disp_l
    dR = disp_r
    yy, xx = torch.meshgrid(
        torch.arange(H, device=dL.device),
        torch.arange(W, device=dL.device),
        indexing="ij",
    )
    tx = xx.unsqueeze(0) - dL
    grid = torch.stack(
        [2.0 * tx / max(W - 1, 1) - 1.0,
         2.0 * yy.unsqueeze(0).expand_as(tx) / max(H - 1, 1) - 1.0],
        dim=-1,
    )
    dR_at_l = F.grid_sample(
        dR.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    diff = (dL - dR_at_l).abs()
    return diff, tx


def _lr_consistency_mask(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """左右一致性：|dL - dR(x-dL)| < max(1.0, 5% * dL) 且 x-dL >= 0。"""
    diff, tx = _lr_error(disp_l, disp_r, H, W)
    dL = disp_l[0]
    thresh = torch.maximum(torch.full_like(dL, 1.0), 0.05 * dL)
    return ((diff < thresh) & (tx >= 0)).float()


def _lr_confidence(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """平滑的左右一致性置信度：exp(-err/thresh)，右图外为 0。"""
    diff, tx = _lr_error(disp_l, disp_r, H, W)
    dL = disp_l[0]
    thresh = torch.maximum(torch.full_like(dL, 1.0), 0.05 * dL)
    conf = torch.exp(-diff / thresh.clamp_min(1e-3))
    return conf * (tx >= 0).float()


def _lr_consistency_mask_batch(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    diff, tx = _lr_error_batch(disp_l, disp_r, H, W)
    thresh = torch.maximum(torch.ones_like(disp_l), 0.05 * disp_l)
    return ((diff < thresh) & (tx >= 0)).float()


def _lr_confidence_batch(
    disp_l: torch.Tensor, disp_r: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    diff, tx = _lr_error_batch(disp_l, disp_r, H, W)
    thresh = torch.maximum(torch.ones_like(disp_l), 0.05 * disp_l)
    conf = torch.exp(-diff / thresh.clamp_min(1e-3))
    return conf * (tx >= 0).float()


def _visibility_mask_batch(disp: torch.Tensor, H: int, W: int) -> torch.Tensor:
    _, xx = torch.meshgrid(
        torch.arange(H, device=disp.device),
        torch.arange(W, device=disp.device),
        indexing="ij",
    )
    return (xx.unsqueeze(0) - disp >= 0).float()


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


def _conf_from_info_batch(info: torch.Tensor, conf_mode: str, disp: torch.Tensor) -> torch.Tensor:
    if conf_mode == "ones":
        return torch.ones_like(disp)
    return info[:, :2].softmax(dim=1)[:, 0]


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
    timing_out: dict | None = None,
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
    if conf_mode not in ("info", "ones", "lr"):
        raise ValueError(f"未知置信度模式: {conf_mode}（可选 info/ones/lr）")
    if occ_mode not in ("lr", "visibility"):
        raise ValueError(f"未知遮挡模式: {occ_mode}（可选 lr/visibility）")

    recorder = _TimingRecorder(device) if timing_out is not None else None
    t_total = time.perf_counter()
    if recorder is not None:
        with recorder.measure("input_transfer"):
            lt = left.to(device)
            rt = right.to(device)
    else:
        lt = left.to(device)
        rt = right.to(device)
    need_r = (occ_mode == "lr") or (conf_mode == "lr")
    if need_r:
        # 正值约束模型：右参考必须水平翻转+交换输入，否则 dR 为半尺度垃圾值。
        out_bi, batch, elapsed = _run_bidirectional_once(
            model, lt, rt, hiera_mode, use_amp, timing=recorder
        )
        if recorder is not None:
            with recorder.measure("output_split_flip"):
                disp = out_bi["disp_pred"][:batch]
                dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2])
                info = out_bi["delta_info_preds"][-1][:batch]
        else:
            disp = out_bi["disp_pred"][:batch]
            dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2])
            info = out_bi["delta_info_preds"][-1][:batch]
    else:
        out_l, elapsed = _run_once(model, lt, rt, hiera_mode, use_amp, timing=recorder)
        disp = out_l["disp_pred"]
        info = out_l["delta_info_preds"][-1]
    if recorder is not None:
        post_context = recorder.measure("confidence_occ_postprocess")
    else:
        post_context = nullcontext()
    with post_context:
        if occ_mode == "lr":
            occ = _lr_consistency_mask(disp, dR, H, W)
        else:
            occ = _visibility_mask(disp, H, W)
        if conf_mode == "lr":
            conf = _lr_confidence(disp, dR, H, W)
        else:
            conf = _conf_from_info(info, conf_mode, disp)

    if recorder is not None:
        with recorder.measure("output_cpu_transfer"):
            result = disp[0].float().cpu(), occ.float().cpu(), conf.float().cpu(), elapsed
        recorder._sync()
        timing_out.update(recorder.stages)
        timing_out["waft_total_seconds"] = time.perf_counter() - t_total
        timing_out["model_forward_seconds"] = recorder.stages.get("model_forward", elapsed)
    else:
        result = disp[0].float().cpu(), occ.float().cpu(), conf.float().cpu(), elapsed
    return result


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
    timing_out: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """双向视差推理（左右参考各一次前向）。

    WAFT 等正值约束模型直接交换 (right, left) 输入会得到错误的半尺度视差
    （实测 dR≈dL/2、LR 一致性 ~1%）；正确做法是水平翻转两张图后再交换
    输入，让视差恢复为正，最后把输出翻回原坐标。
    与 run_stereo_matching(occ_mode="lr") 同样两次前向，额外返回真实 dR。

    Returns:
        (dL, dR, occL, occR, confL, confR, elapsed)，均为 (H, W) float32 CPU。
    """
    H, W = left.shape[-2:]
    hiera_mode = _resolve_hiera(hiera, H, W)
    if hiera_mode not in ("direct", "hiera"):
        raise ValueError(f"未知推理模式: {hiera}（可选 auto/direct/hiera）")
    if conf_mode not in ("info", "ones", "lr"):
        raise ValueError(f"未知置信度模式: {conf_mode}（可选 info/ones/lr）")
    if occ_mode not in ("lr", "visibility"):
        raise ValueError(f"未知遮挡模式: {occ_mode}（可选 lr/visibility）")

    recorder = _TimingRecorder(device) if timing_out is not None else None
    t_total = time.perf_counter()
    if recorder is not None:
        with recorder.measure("input_transfer"):
            lt = left.to(device)
            rt = right.to(device)
    else:
        lt = left.to(device)
        rt = right.to(device)
    out_bi, batch, elapsed = _run_bidirectional_once(
        model, lt, rt, hiera_mode, use_amp, timing=recorder
    )
    if recorder is not None:
        with recorder.measure("output_split_flip"):
            dL = out_bi["disp_pred"][:batch]
            dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2])
            infoL = out_bi["delta_info_preds"][-1][:batch]
            infoR = torch.flip(out_bi["delta_info_preds"][-1][batch:], dims=[3])
    else:
        dL = out_bi["disp_pred"][:batch]
        dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2])  # 翻回原右图坐标
        infoL = out_bi["delta_info_preds"][-1][:batch]
        infoR = torch.flip(out_bi["delta_info_preds"][-1][batch:], dims=[3])
    if recorder is not None:
        post_context = recorder.measure("confidence_occ_postprocess")
    else:
        post_context = nullcontext()
    with post_context:
        if occ_mode == "visibility":
            occL = _visibility_mask(dL, H, W)
            occR = _visibility_mask(dR, H, W)
        else:
            occL = _lr_consistency_mask(dL, dR, H, W)
            occR = _lr_consistency_mask(dR, dL, H, W)
        if conf_mode == "lr":
            confL = _lr_confidence(dL, dR, H, W)
            confR = _lr_confidence(dR, dL, H, W)
        else:
            confL = _conf_from_info(infoL, conf_mode, dL)
            confR = _conf_from_info(infoR, conf_mode, dR)
    if recorder is not None:
        with recorder.measure("output_cpu_transfer"):
            result = (
                dL[0].float().cpu(), dR[0].float().cpu(), occL.float().cpu(),
                occR.float().cpu(), confL.float().cpu(), confR.float().cpu(), elapsed,
            )
        recorder._sync()
        timing_out.update(recorder.stages)
        timing_out["waft_total_seconds"] = time.perf_counter() - t_total
        timing_out["model_forward_seconds"] = recorder.stages.get("model_forward", elapsed)
    else:
        result = (
            dL[0].float().cpu(), dR[0].float().cpu(), occL.float().cpu(),
            occR.float().cpu(), confL.float().cpu(), confR.float().cpu(), elapsed,
        )
    return result


@torch.no_grad()
def run_stereo_matching_batch(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    hiera: str = "auto",
    conf_mode: str = "info",
    occ_mode: str = "visibility",
    timing_out: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """批量单向左参考视差推理（左相机视角深度视频用）。

    left/right: (B, 3, H, W) 0-255 float RGB。

    Returns:
        disp/occ/conf: (B, H, W) float32 CPU tensor；
        elapsed: 模型前向耗时。
    """
    H, W = left.shape[-2:]
    hiera_mode = _resolve_hiera(hiera, H, W)
    if hiera_mode not in ("direct", "hiera"):
        raise ValueError(f"未知推理模式: {hiera}（可选 auto/direct/hiera）")
    if conf_mode not in ("info", "ones"):
        raise ValueError("单向 batch 推理 conf_mode 仅支持 info/ones")
    if occ_mode != "visibility":
        raise ValueError("单向 batch 推理 occ_mode 仅支持 visibility")

    recorder = _TimingRecorder(device) if timing_out is not None else None
    t_total = time.perf_counter()
    if recorder is not None:
        with recorder.measure("input_transfer"):
            lt = left.to(device)
            rt = right.to(device)
    else:
        lt = left.to(device)
        rt = right.to(device)

    out_l, elapsed = _run_once(model, lt, rt, hiera_mode, use_amp, timing=recorder)
    if recorder is not None:
        split_context = recorder.measure("output_split_flip")
    else:
        split_context = nullcontext()
    with split_context:
        disp = out_l["disp_pred"].float()
        info = out_l["delta_info_preds"][-1]

    if recorder is not None:
        post_context = recorder.measure("confidence_occ_postprocess")
    else:
        post_context = nullcontext()
    with post_context:
        occ = _visibility_mask_batch(disp, H, W)
        conf = _conf_from_info_batch(info, conf_mode, disp)

    if recorder is not None:
        with recorder.measure("output_cpu_transfer"):
            result = disp.cpu(), occ.cpu(), conf.cpu(), elapsed
        recorder._sync()
        timing_out.update(recorder.stages)
        timing_out["waft_total_seconds"] = time.perf_counter() - t_total
        timing_out["model_forward_seconds"] = recorder.stages.get("model_forward", elapsed)
    else:
        result = disp.cpu(), occ.cpu(), conf.cpu(), elapsed
    return result


@torch.no_grad()
def run_stereo_matching_bi_batch(
    model: WAFT,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    hiera: str = "auto",
    conf_mode: str = "lr",
    occ_mode: str = "lr",
    timing_out: dict | None = None,
    temporal_flow_model: torch.nn.Module | None = None,
    temporal_state: dict | None = None,
    temporal_flow_iters: int = 12,
    temporal_blend: float = 0.75,
    temporal_photo_tol: float = 40.0,
    temporal_flow_abs_tol: float = 0.5,
    temporal_flow_rel_tol: float = 0.01,
    temporal_disp_abs_tol: float = 3.0,
    temporal_disp_rel_tol: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """批量双向视差推理（深度视频用）。

    left/right: (B, 3, H, W) 0-255 float RGB。模型前向按 batch 一次完成，
    LR 一致性/置信度按帧逐帧计算（现有辅助函数为单帧实现）。

    Returns:
        (dL, dR, occL, occR, confL, confR, elapsed)，均为 (B, H, W) float32 CPU。
    """
    H, W = left.shape[-2:]
    hiera_mode = _resolve_hiera(hiera, H, W)
    if hiera_mode not in ("direct", "hiera"):
        raise ValueError(f"未知推理模式: {hiera}（可选 auto/direct/hiera）")
    if conf_mode not in ("info", "ones", "lr"):
        raise ValueError(f"未知置信度模式: {conf_mode}（可选 info/ones/lr）")
    if occ_mode not in ("lr", "visibility"):
        raise ValueError(f"未知遮挡模式: {occ_mode}（可选 lr/visibility）")
    recorder = _TimingRecorder(device) if timing_out is not None else None
    t_total = time.perf_counter()
    if recorder is not None:
        with recorder.measure("input_transfer"):
            lt = left.to(device)
            rt = right.to(device)
    else:
        lt = left.to(device)
        rt = right.to(device)
    temporal_init = None
    if temporal_flow_model is not None:
        if hiera_mode != "direct":
            raise ValueError("WAFT temporal initialization supports direct inference only")
        if temporal_state is None:
            raise ValueError("temporal_state is required with temporal_flow_model")
        from stereo_center.raft_flow import flow_between

        if recorder is not None:
            temporal_context = recorder.measure("temporal_flow")
        else:
            temporal_context = nullcontext()
        with temporal_context:
            current_refs, previous_refs, boundary_mask = _pack_temporal_reference_pairs(
                lt, rt, temporal_state
            )
            backward_flow = flow_between(
                temporal_flow_model,
                current_refs,
                previous_refs,
                iters=int(temporal_flow_iters),
            )
            forward_flow = flow_between(
                temporal_flow_model,
                previous_refs,
                current_refs,
                iters=int(temporal_flow_iters),
            )
            temporal_mask = temporal_alignment_mask(
                current_refs,
                previous_refs,
                forward_flow,
                backward_flow,
                photo_tol=float(temporal_photo_tol),
                flow_abs_tol=float(temporal_flow_abs_tol),
                flow_rel_tol=float(temporal_flow_rel_tol),
            ) * boundary_mask
        temporal_init = {
            "backward_flow": backward_flow,
            "mask": temporal_mask,
            "group_size": left.shape[0],
            "carry_disparity": temporal_state.get("previous_disparities"),
            "blend": float(temporal_blend),
            "disparity_abs_tol": float(temporal_disp_abs_tol),
            "disparity_rel_tol": float(temporal_disp_rel_tol),
        }
    out_bi, batch, elapsed = _run_bidirectional_once(
        model, lt, rt, hiera_mode, use_amp, timing=recorder,
        temporal_init=temporal_init,
    )
    if temporal_flow_model is not None:
        temporal_state["previous_reference_images"] = torch.stack(
            (lt[-1], torch.flip(rt[-1], dims=[2])), dim=0
        ).detach()
        temporal_state["previous_disparities"] = torch.stack(
            (out_bi["disp_pred"][batch - 1], out_bi["disp_pred"][2 * batch - 1]),
            dim=0,
        ).detach()
        if timing_out is not None and "temporal_valid_mask" in out_bi:
            timing_out["temporal_valid_ratio"] = float(
                out_bi["temporal_valid_mask"].float().mean().item()
            )
    if recorder is not None:
        with recorder.measure("output_split_flip"):
            dL = out_bi["disp_pred"][:batch].float()
            dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2]).float()
            infoL = out_bi["delta_info_preds"][-1][:batch]
            infoR = torch.flip(out_bi["delta_info_preds"][-1][batch:], dims=[3])
    else:
        dL = out_bi["disp_pred"][:batch].float()
        dR = torch.flip(out_bi["disp_pred"][batch:], dims=[2]).float()
        infoL = out_bi["delta_info_preds"][-1][:batch]
        infoR = torch.flip(out_bi["delta_info_preds"][-1][batch:], dims=[3])
    if recorder is not None:
        post_context = recorder.measure("confidence_occ_postprocess")
    else:
        post_context = nullcontext()
    with post_context:
        if occ_mode == "visibility":
            occL = _visibility_mask_batch(dL, H, W)
            occR = _visibility_mask_batch(dR, H, W)
        else:
            occL = _lr_consistency_mask_batch(dL, dR, H, W)
            occR = _lr_consistency_mask_batch(dR, dL, H, W)
        if conf_mode == "lr":
            confL = _lr_confidence_batch(dL, dR, H, W)
            confR = _lr_confidence_batch(dR, dL, H, W)
        else:
            confL = _conf_from_info_batch(infoL, conf_mode, dL)
            confR = _conf_from_info_batch(infoR, conf_mode, dR)
    if recorder is not None:
        with recorder.measure("output_cpu_transfer"):
            result = dL.cpu(), dR.cpu(), occL.cpu(), occR.cpu(), confL.cpu(), confR.cpu(), elapsed
        recorder._sync()
        timing_out.update(recorder.stages)
        timing_out["waft_total_seconds"] = time.perf_counter() - t_total
        timing_out["model_forward_seconds"] = recorder.stages.get("model_forward", elapsed)
    else:
        result = dL.cpu(), dR.cpu(), occL.cpu(), occR.cpu(), confL.cpu(), confR.cpu(), elapsed
    return result
