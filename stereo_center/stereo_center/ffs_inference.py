"""Fast Foundation Stereo inference wrapper.

The wrapper follows the same shape as the other stereo backends:
    load_ffs(...) -> model
    run_stereo_matching(...) -> (disp, occ, conf, elapsed)
    run_stereo_matching_bi_batch(...) -> (dL, dR, occL, occR, confL, confR, elapsed)

Heavy FFS dependencies are imported lazily so backend registration and path tests do
not require a local FFS checkout.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # stereo_center/
DEFAULT_CKPT_NAME = "model_best_bp2_serialize.pth"


@dataclass
class FFSModel:
    model: torch.nn.Module
    max_disp: int
    valid_iters: int


def resolve_checkpoint(weights: str | Path) -> Path:
    """Return the FFS checkpoint path from a directory or a direct file path."""
    path = Path(weights).expanduser()
    if path.is_file():
        return path
    ckpt = path / DEFAULT_CKPT_NAME
    if ckpt.exists():
        return ckpt
    raise FileNotFoundError(
        f"FFS checkpoint not found: {ckpt}\n"
        f"Pass --weights as either a directory containing {DEFAULT_CKPT_NAME} "
        f"or a direct .pth checkpoint path."
    )


def resolve_ffs_root(explicit: str | Path | None = None) -> Path:
    """Locate the Fast-FoundationStereo source tree and put it first on sys.path."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("FFS_ROOT"):
        candidates.append(Path(os.environ["FFS_ROOT"]).expanduser())
    candidates += [
        PROJECT_ROOT / "third_party" / "Fast-FoundationStereo",
        PROJECT_ROOT.parent / "Fast-FoundationStereo",
        Path.home() / "BothEyesDepth" / "Fast-FoundationStereo",
    ]
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "core" / "utils" / "utils.py").exists():
            root_s = str(root)
            if sys.path[:1] != [root_s]:
                sys.path = [p for p in sys.path if p != root_s]
                sys.path.insert(0, root_s)
            _clear_foreign_core_modules(root)
            return root
    raise FileNotFoundError(
        "Fast-FoundationStereo source tree not found. "
        "Clone https://github.com/NVlabs/Fast-FoundationStereo and pass --ffs-root "
        "or set FFS_ROOT."
    )


def _clear_foreign_core_modules(ffs_root: Path) -> None:
    """Avoid collisions with other third-party packages that also import as `core`."""
    stale = []
    for name, module in list(sys.modules.items()):
        if name != "core" and not name.startswith("core."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve().relative_to(ffs_root)
        except ValueError:
            stale.append(name)
    for name in stale:
        sys.modules.pop(name, None)


def _set_model_args(model: torch.nn.Module, max_disp: int, valid_iters: int) -> None:
    args = getattr(model, "args", None)
    if args is None:
        args = SimpleNamespace()
        setattr(model, "args", args)
    setattr(args, "max_disp", int(max_disp))
    setattr(args, "valid_iters", int(valid_iters))


def load_ffs(
    model_type: str = "default",
    weights_dir: str | Path = "weights/pretrain_weights/20-26-39",
    device: str = "cuda",
    max_disp: int = 416,
    valid_iters: int = 8,
    ffs_root: str | Path | None = None,
) -> FFSModel:
    """Load a serialized Fast Foundation Stereo model checkpoint."""
    resolve_ffs_root(ffs_root)
    ckpt_path = resolve_checkpoint(weights_dir)
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Unsupported FFS checkpoint payload in {ckpt_path}: {type(model).__name__}. "
            "Expected a serialized torch.nn.Module or a dict with key 'model'."
        )
    _set_model_args(model, max_disp, valid_iters)
    model.eval().to(device)
    print(
        f"[ffs] Fast Foundation Stereo loaded: {ckpt_path} "
        f"(model_type={model_type}, max_disp={max_disp}, valid_iters={valid_iters})"
    )
    return FFSModel(model=model, max_disp=int(max_disp), valid_iters=int(valid_iters))


def _padder(shape):
    from core.utils.utils import InputPadder

    try:
        return InputPadder(shape, divis_by=32, force_square=False)
    except TypeError:
        return InputPadder(shape, divis_by=32)


def _extract_disparity(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("disp", "disp_pred", "flow_pr", "flow"):
            if key in output:
                output = output[key]
                break
    elif isinstance(output, (tuple, list)):
        output = output[-1]
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if not torch.is_tensor(output):
        raise TypeError(f"FFS forward returned unsupported output: {type(output).__name__}")
    disp = output.float()
    if disp.ndim == 4:
        if disp.shape[1] == 1:
            disp = disp[:, 0]
        elif disp.shape[1] == 2:
            disp = disp[:, 0]
        else:
            raise ValueError(f"Unexpected FFS disparity shape: {tuple(disp.shape)}")
    if disp.ndim != 3:
        raise ValueError(f"Unexpected FFS disparity shape: {tuple(disp.shape)}")
    return disp.abs()


def _forward(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    valid_iters: int,
    low_memory: bool,
) -> torch.Tensor:
    attempts = [
        dict(
            iters=valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
            low_memory=low_memory,
        ),
        dict(iters=valid_iters, test_mode=True, low_memory=low_memory),
        dict(test_mode=True),
        {},
    ]
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return _extract_disparity(model.forward(left, right, **kwargs))
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _run_once(wrapper: FFSModel, left: torch.Tensor, right: torch.Tensor, use_amp: bool) -> tuple[torch.Tensor, float]:
    padder = _padder(left.shape)
    left_pad, right_pad = padder.pad(left, right)
    left_pad = left_pad.contiguous()
    right_pad = right_pad.contiguous()
    low_memory = left_pad.shape[0] > 4
    if left_pad.is_cuda:
        torch.cuda.synchronize(left_pad.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_amp and left_pad.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                disp = _forward(wrapper.model, left_pad, right_pad, wrapper.valid_iters, low_memory)
        else:
            disp = _forward(wrapper.model, left_pad, right_pad, wrapper.valid_iters, low_memory)
    if left_pad.is_cuda:
        torch.cuda.synchronize(left_pad.device)
    elapsed = time.perf_counter() - t0
    disp = padder.unpad(disp.unsqueeze(1))[:, 0].float().cpu()
    return disp, elapsed


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    _, _, W = disp.shape
    x = torch.arange(W, device=disp.device).view(1, 1, W)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < W - 1)).float()


def _lr_consistency(dL: torch.Tensor, dR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, W = dL.shape
    yy = torch.linspace(-1, 1, H, device=dL.device).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=dL.device).float()
    xgrid = xx.view(1, 1, W)
    grid_l = torch.stack([((xgrid - dL) / max(W - 1, 1) * 2 - 1).clamp(-1, 1), yy], dim=-1)
    dR_at_L = F.grid_sample(dR.unsqueeze(1), grid_l, mode="bilinear", align_corners=True)[:, 0]
    thresh_l = torch.maximum(torch.ones_like(dL), 0.05 * dL)
    occL = ((dL - dR_at_L).abs() <= thresh_l) & (xgrid - dL >= 0)

    grid_r = torch.stack([((xgrid - dR) / max(W - 1, 1) * 2 - 1).clamp(-1, 1), yy], dim=-1)
    dL_at_R = F.grid_sample(dL.unsqueeze(1), grid_r, mode="bilinear", align_corners=True)[:, 0]
    thresh_r = torch.maximum(torch.ones_like(dR), 0.05 * dR)
    occR = ((dR - dL_at_R).abs() <= thresh_r) & (xgrid - dR >= 0)
    return occL.float(), occR.float()


def _lr_confidence(dL: torch.Tensor, dR: torch.Tensor) -> torch.Tensor:
    """Smooth confidence from left-right disparity agreement.

    For each reference pixel, sample the opposite-view disparity at x - d and
    convert the absolute mismatch to exp(-err / max(1px, 5% disparity)).
    Pixels whose correspondence leaves the image receive confidence 0.
    """
    B, H, W = dL.shape
    yy = torch.linspace(-1, 1, H, device=dL.device).view(1, H, 1).expand(B, H, W)
    xx = torch.arange(W, device=dL.device).float()
    xgrid = xx.view(1, 1, W)
    x_match = xgrid - dL
    grid = torch.stack([((x_match) / max(W - 1, 1) * 2 - 1).clamp(-1, 1), yy], dim=-1)
    dR_at_L = F.grid_sample(dR.unsqueeze(1), grid, mode="bilinear", align_corners=True)[:, 0]
    thresh = torch.maximum(torch.ones_like(dL), 0.05 * dL)
    conf = torch.exp(-(dL - dR_at_L).abs() / thresh.clamp_min(1e-3))
    return conf * (x_match >= 0).float()


@torch.no_grad()
def run_stereo_matching(
    model: FFSModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    conf_mode: str = "ones",
    occ_mode: str = "visibility",
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    d, elapsed = _run_once(model, left.to(device), right.to(device), use_amp)
    occ = _visibility(d)
    conf = torch.ones_like(d)
    if d.shape[0] == 1:
        return d[0], occ[0], conf[0], elapsed
    return d, occ, conf, elapsed


@torch.no_grad()
def run_stereo_matching_bi_batch(
    model: FFSModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    use_amp: bool = True,
    conf_mode: str = "ones",
    occ_mode: str = "visibility",
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lt = left.to(device)
    rt = right.to(device)
    packed_left = torch.cat((lt, torch.flip(rt, dims=[3])), dim=0).contiguous()
    packed_right = torch.cat((rt, torch.flip(lt, dims=[3])), dim=0).contiguous()
    disp_bi, elapsed = _run_once(model, packed_left, packed_right, use_amp)
    batch = left.shape[0]
    dL = disp_bi[:batch]
    dR = torch.flip(disp_bi[batch:], dims=[2])
    if occ_mode == "lr":
        occL, occR = _lr_consistency(dL.to(device), dR.to(device))
        occL, occR = occL.cpu(), occR.cpu()
    else:
        occL = _visibility(dL)
        occR = _visibility(dR)
    if conf_mode == "lr":
        confL = _lr_confidence(dL.to(device), dR.to(device)).cpu()
        confR = _lr_confidence(dR.to(device), dL.to(device)).cpu()
    else:
        confL = torch.ones_like(dL)
        confR = torch.ones_like(dR)
    return dL, dR, occL, occR, confL, confR, elapsed


@torch.no_grad()
def run_stereo_matching_bi(
    model: FFSModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    out = run_stereo_matching_bi_batch(model, left, right, device, **kwargs)
    return tuple(x[0] for x in out[:6]) + (out[6],)
