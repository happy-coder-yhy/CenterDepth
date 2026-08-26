"""Inference adapter for the pinned third-party StereoNet implementation."""

from __future__ import annotations

import importlib
import hashlib
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


CHECKPOINT_NAME = "epoch=20-step=744533.ckpt"
PINNED_SOURCE_REVISION = "9c0260f270547d8001e9d637cf3a94658f805bae"
CHECKPOINT_SHA256 = "03b67d8571f39505959cf485de272fe0ea615a1d8dd3fab16f06af4acec2b82e"
DEFAULT_MAX_SIDE = 625
STRIDE = 8


def resolve_stereonet_root(explicit: str | Path | None = None) -> Path:
    """Find a StereoNet checkout containing its importable ``src`` directory."""
    default_root = Path(__file__).resolve().parents[1] / "third_party" / "StereoNet_PyTorch"
    candidates = (explicit, os.environ.get("STEREONET_ROOT"), default_root)
    checked: list[Path] = []
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        root = Path(candidate).expanduser()
        checked.append(root)
        if (root / "src" / "stereonet" / "model.py").is_file():
            return root
    attempted = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "StereoNet source root was not found. Expected "
        "<root>/src/stereonet/model.py; checked: " + attempted
    )


def resolve_checkpoint(weights: str | Path) -> Path:
    """Resolve the known checkpoint from either its path or its parent directory."""
    path = Path(weights).expanduser()
    if path.is_file():
        return path
    expected = path / CHECKPOINT_NAME if path.is_dir() else path
    if expected.is_file():
        return expected
    raise FileNotFoundError(
        "StereoNet checkpoint was not found. Pass either the checkpoint file or "
        f"a directory containing {CHECKPOINT_NAME}: {path}"
    )


def normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove exactly one Lightning ``model.`` prefix from checkpoint keys."""
    return {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
    }


def _validate_stereo_inputs(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 4 or right.ndim != 4:
        raise ValueError("StereoNet inputs must be Bx3xHxW tensors")
    if left.shape != right.shape or left.shape[1] != 3:
        raise ValueError("StereoNet left/right inputs must have matching Bx3xHxW shapes")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("StereoNet inputs must be floating point RGB tensors in 0..255")


def prepare_inputs(
    left: torch.Tensor,
    right: torch.Tensor,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Resize a pair together and convert pipeline RGB values to StereoNet RGB."""
    _validate_stereo_inputs(left, right)
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    source_h, source_w = left.shape[-2:]
    longest = max(source_h, source_w)
    scale = min(1.0, max_side / longest)
    target_h = max(1, int(source_h * scale))
    target_w = max(1, int(source_w * scale))
    if (target_h, target_w) != (source_h, source_w):
        left = F.interpolate(left, size=(target_h, target_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # The upstream ToTensor transform yields [0, 1], then Rescale maps to [-1, 1].
    return left.div(255.0).sub(0.5).mul(2.0), right.div(255.0).sub(0.5).mul(2.0), target_w / source_w


def _pad_to_stride(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = tensor.shape[-2:]
    pad_bottom = (-height) % STRIDE
    pad_right = (-width) % STRIDE
    return F.pad(tensor, (0, pad_right, 0, pad_bottom)), pad_bottom, pad_right


def restore_disparity(
    disparity: torch.Tensor,
    output_hw: tuple[int, int],
    scale_x: float,
) -> torch.Tensor:
    """Resize disparity into the source pixel grid and restore its pixel units."""
    if disparity.ndim != 3:
        raise ValueError("disparity must have shape BxHxW")
    if scale_x <= 0:
        raise ValueError("scale_x must be positive")
    restored = F.interpolate(
        disparity.unsqueeze(1), size=output_hw, mode="bilinear", align_corners=False
    ).squeeze(1)
    return restored / scale_x


def _module_is_from_source(module: types.ModuleType, source_dir: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        Path(module_file).resolve().relative_to(source_dir.resolve())
    except ValueError:
        return False
    return True


def _install_lightning_compat() -> bool:
    """Install only the tiny Lightning surface needed to import the upstream model."""
    try:
        importlib.import_module("pytorch_lightning")
        return False
    except ModuleNotFoundError as error:
        if error.name != "pytorch_lightning":
            raise

    lightning = types.ModuleType("pytorch_lightning")

    class LightningModule(nn.Module):
        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            return None

    LightningModule.__module__ = "pytorch_lightning"
    lightning.LightningModule = LightningModule
    sys.modules["pytorch_lightning"] = lightning
    return True


def _compat_model_checkpoint() -> type:
    """Supply the metadata-only class needed by PyTorch's weights-only unpickler."""
    callbacks_name = "pytorch_lightning.callbacks"
    checkpoint_name = f"{callbacks_name}.model_checkpoint"
    callbacks = sys.modules.get(callbacks_name)
    if callbacks is None:
        callbacks = types.ModuleType(callbacks_name)
        sys.modules[callbacks_name] = callbacks
    module = sys.modules.get(checkpoint_name)
    if module is not None and hasattr(module, "ModelCheckpoint"):
        return module.ModelCheckpoint

    module = types.ModuleType(checkpoint_name)

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = checkpoint_name
    module.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = module
    sys.modules[checkpoint_name] = module
    return ModelCheckpoint


@contextmanager
def _checkpoint_safe_globals() -> Iterator[None]:
    """Allowlist only Lightning checkpoint metadata for a weights-only load."""
    try:
        model_checkpoint = importlib.import_module(
            "pytorch_lightning.callbacks.model_checkpoint"
        ).ModelCheckpoint
    except ModuleNotFoundError:
        model_checkpoint = _compat_model_checkpoint()
    with torch.serialization.safe_globals([model_checkpoint]):
        yield


def _missing_skimage_in_upstream_utils(error: ModuleNotFoundError, source_dir: Path) -> bool:
    if error.name != "skimage":
        return False
    utils_path = (source_dir / "stereonet" / "utils.py").resolve()
    traceback = error.__traceback__
    while traceback is not None:
        try:
            if Path(traceback.tb_frame.f_code.co_filename).resolve() == utils_path:
                return True
        except OSError:
            pass
        traceback = traceback.tb_next
    return False


def _install_inference_utils_compat() -> None:
    """Provide only the validation-only utility symbol required at model import time."""
    utils_module = types.ModuleType("stereonet.utils")

    def plot_figure(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("StereoNet plotting requires the optional scikit-image dependency")

    utils_module.plot_figure = plot_figure
    sys.modules["stereonet.utils"] = utils_module


def _is_upstream_compat_module(name: str) -> bool:
    return name == "stereonet" or name.startswith("stereonet.") or name == "pytorch_lightning" or name.startswith("pytorch_lightning.")


@contextmanager
def _upstream_import_transaction(source_root: Path) -> Iterator[None]:
    """Temporarily expose upstream import dependencies without mutating process state."""
    source_dir = source_root / "src"
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if _is_upstream_compat_module(name)
    }
    try:
        loaded = sys.modules.get("stereonet")
        if loaded is not None and not _module_is_from_source(loaded, source_dir):
            for name in tuple(sys.modules):
                if name == "stereonet" or name.startswith("stereonet."):
                    del sys.modules[name]
        sys.path.insert(0, str(source_dir))
        _install_lightning_compat()
        yield
    finally:
        for name in tuple(sys.modules):
            if _is_upstream_compat_module(name) and name not in saved_modules:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _import_upstream_model(source_root: Path):
    source_dir = source_root / "src"
    try:
        return importlib.import_module("stereonet.model")
    except ModuleNotFoundError as error:
        if not _missing_skimage_in_upstream_utils(error, source_dir):
            raise
        sys.modules.pop("stereonet.model", None)
        sys.modules.pop("stereonet.utils", None)
        _install_inference_utils_compat()
        return importlib.import_module("stereonet.model")


def _source_revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def validate_source_revision(source_root: Path) -> str:
    """Require the exact upstream revision recorded for this adapter."""
    revision = _source_revision(source_root)
    if revision != PINNED_SOURCE_REVISION:
        raise RuntimeError(
            "StereoNet source revision mismatch: "
            f"expected {PINNED_SOURCE_REVISION}, got {revision}"
        )
    return revision


def verify_checkpoint_sha256(checkpoint: Path) -> str:
    """Verify the downloaded checkpoint before any deserialization occurs."""
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            "StereoNet checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual}"
        )
    return actual


@dataclass
class StereoNetModel:
    model: nn.Module
    checkpoint: Path
    source_root: Path
    source_revision: str
    max_side: int
    soft_argmin: Callable[[torch.Tensor, int], torch.Tensor]
    timing: dict[str, float] = field(default_factory=dict)


def load_stereonet(
    model_type: str = "stereonet_sceneflow_rgb",
    weights_dir: str | Path = "weights/stereonet",
    device: str = "cpu",
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    stereonet_root: str | Path | None = None,
) -> StereoNetModel:
    """Construct the pinned architecture and load its checkpoint strictly."""
    if model_type != "stereonet_sceneflow_rgb":
        raise ValueError(
            "unknown StereoNet model type: "
            f"{model_type} (accepted: stereonet_sceneflow_rgb)"
        )
    source_root = resolve_stereonet_root(stereonet_root)
    checkpoint = resolve_checkpoint(weights_dir)
    source_revision = validate_source_revision(source_root)
    verify_checkpoint_sha256(checkpoint)
    with _upstream_import_transaction(source_root):
        upstream = _import_upstream_model(source_root)
        model = upstream.StereoNet(
            k_downsampling_layers=3,
            k_refinement_layers=3,
            candidate_disparities=256,
            feature_extractor_filters=32,
            cost_volumizer_filters=32,
        )
        with _checkpoint_safe_globals():
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint_data["state_dict"] if "state_dict" in checkpoint_data else checkpoint_data
    model.load_state_dict(normalize_state_dict_keys(state_dict), strict=True)
    model.eval().to(device)
    return StereoNetModel(
        model=model,
        checkpoint=checkpoint,
        source_root=source_root,
        source_revision=source_revision,
        max_side=max_side,
        soft_argmin=upstream.soft_argmin,
    )


def _synchronize(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


@contextmanager
def _measure(timing: dict[str, float], name: str, device: str | torch.device) -> Iterator[None]:
    _synchronize(device)
    started = time.perf_counter()
    yield
    _synchronize(device)
    timing[name] = time.perf_counter() - started


@torch.no_grad()
def run_stereo_matching(
    wrapper: StereoNetModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cpu",
    timing_out: dict[str, float] | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Run the actual StereoNet stages and recover source-grid disparity."""
    source_hw = tuple(left.shape[-2:])
    timing: dict[str, float] = {}
    _synchronize(device)
    total_started = time.perf_counter()
    with _measure(timing, "input_resize_normalize_seconds", device):
        prepared_left, prepared_right, scale_x = prepare_inputs(left, right, wrapper.max_side)
    prepared_hw = tuple(prepared_left.shape[-2:])
    with _measure(timing, "input_pad_seconds", device):
        padded_left, pad_bottom, pad_right = _pad_to_stride(prepared_left)
        padded_right, right_pad_bottom, right_pad_right = _pad_to_stride(prepared_right)
        if (pad_bottom, pad_right) != (right_pad_bottom, right_pad_right):
            raise RuntimeError("StereoNet pair padding must match")
    with _measure(timing, "device_transfer_seconds", device):
        padded_left = padded_left.to(device)
        padded_right = padded_right.to(device)

    upstream = wrapper.model
    with _measure(timing, "stereo_forward_seconds", device):
        with _measure(timing, "feature_extraction_seconds", device):
            left_embedding = upstream.feature_extractor(padded_left)
            right_embedding = upstream.feature_extractor(padded_right)
        with _measure(timing, "cost_volume_seconds", device):
            cost = upstream.cost_volumizer((left_embedding, right_embedding), side="left")
        with _measure(timing, "coarse_regression_seconds", device):
            coarse = wrapper.soft_argmin(cost, upstream.candidate_disparities)
        disparity = coarse
        for index, refiner in enumerate(upstream.refiners, start=1):
            with _measure(timing, f"refinement_{index}_seconds", device):
                scale = (2**upstream.k_refinement_layers) / (2**index)
                height = int(padded_left.size(2) // scale)
                width = int(padded_left.size(3) // scale)
                reference = F.interpolate(
                    padded_left, [height, width], mode="bilinear", align_corners=True
                )
                low_res = F.interpolate(
                    disparity, [height, width], mode="bilinear", align_corners=True
                )
                disparity = F.relu(refiner(torch.cat((reference, low_res), dim=1)) + low_res)
    with _measure(timing, "output_unpad_resize_seconds", device):
        disparity = disparity[:, 0, : prepared_hw[0], : prepared_hw[1]]
        disparity = restore_disparity(disparity, source_hw, scale_x)
        output = disparity.float().cpu()
        horizontal = torch.arange(source_hw[1], dtype=output.dtype).view(1, 1, -1)
        visibility = (
            (output >= 0.5)
            & (horizontal - output >= 0)
            & (output < source_hw[1] - 1)
        ).float()
        confidence = torch.ones_like(output)
    _synchronize(device)
    elapsed = time.perf_counter() - total_started
    timing["stereo_total_seconds"] = elapsed
    timing["model_forward_seconds"] = timing["stereo_forward_seconds"]
    wrapper.timing = timing
    if timing_out is not None:
        timing_out.update(timing)
    return output, visibility, confidence, elapsed
