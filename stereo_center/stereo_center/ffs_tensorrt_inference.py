"""Fixed-shape TensorRT runtime for the official FFS split engines.

The feature and post runners are exported by the upstream FFS repository. The
GWC volume between them remains the existing Triton implementation, matching
the upstream ``TrtRunner`` data flow.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


FEATURE_ENGINE = "feature_runner.engine"
POST_ENGINE = "post_runner.engine"
POST_FEATURE_NAMES = (
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "features_right_04",
    "stem_2x",
)


@dataclass
class FFSTensorRTModel:
    runner: object
    batch_size: int
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    post_batch_size: int
    runtime: str = "tensorrt"


def resolve_engine_paths(engine_dir: str | Path) -> tuple[Path, Path]:
    root = Path(engine_dir).expanduser()
    feature = root / FEATURE_ENGINE
    post = root / POST_ENGINE
    missing = [str(path) for path in (feature, post) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "FFS TensorRT engine directory is incomplete; missing: "
            + ", ".join(missing)
        )
    return feature, post


def pad_static_batch(
    left: torch.Tensor, right: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if left.shape[0] != right.shape[0]:
        raise ValueError("FFS TensorRT left/right batch sizes must match")
    if left.shape[0] > batch_size:
        raise ValueError(
            f"FFS TensorRT engine batch size is {batch_size}, got {left.shape[0]}"
        )
    if left.shape[0] == batch_size:
        return left, right
    extra = batch_size - left.shape[0]
    return (
        torch.cat((left, left[-1:].expand(extra, -1, -1, -1)), dim=0),
        torch.cat((right, right[-1:].expand(extra, -1, -1, -1)), dim=0),
    )


def pad_static_spatial(
    image: torch.Tensor, input_height: int, input_width: int
) -> torch.Tensor:
    height, width = image.shape[-2:]
    if height > input_height or width > input_width:
        raise ValueError(
            f"FFS TensorRT engine expects at most {input_height}x{input_width}, "
            f"got {height}x{width}"
        )
    return F.pad(image, (0, input_width - width, 0, input_height - height))


def run_post_in_chunks(
    runner: object,
    engine: object,
    context: object,
    inputs: dict[str, torch.Tensor],
    batch_limit: int,
) -> torch.Tensor:
    """Run the post engine in fixed-size chunks and concatenate disparity."""
    if batch_limit < 1:
        raise ValueError("FFS TensorRT post-engine batch limit must be positive")
    if not inputs:
        raise ValueError("FFS TensorRT post-engine inputs must not be empty")
    batch = next(iter(inputs.values())).shape[0]
    if batch < 1 or any(tensor.shape[0] != batch for tensor in inputs.values()):
        raise ValueError("FFS TensorRT post-engine inputs must share a positive batch size")

    chunks = []
    for start in range(0, batch, batch_limit):
        stop = min(start + batch_limit, batch)
        output = runner.run_trt(
            engine, context, {name: tensor[start:stop] for name, tensor in inputs.items()}
        )
        if "disp" not in output:
            raise ValueError("FFS TensorRT post engine did not return 'disp'")
        chunks.append(output["disp"])
    return torch.cat(chunks, dim=0)


def engine_batch_limit(engine: object, input_name: str) -> int:
    """Read the maximum supported batch size from TensorRT profile zero."""
    _, _, maximum = engine.get_tensor_profile_shape(input_name, 0)
    batch_limit = int(maximum[0])
    if batch_limit < 1:
        raise ValueError(
            f"FFS TensorRT engine has invalid batch profile for {input_name}: {maximum}"
        )
    return batch_limit


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_ffs_tensorrt(
    engine_dir: str | Path,
    ffs_root: str | Path,
    device: str = "cuda",
    batch_size: int = 12,
) -> FFSTensorRTModel:
    if not str(device).startswith("cuda"):
        raise ValueError("FFS TensorRT runtime requires a CUDA device")
    if batch_size != 12:
        raise ValueError("The fixed FFS TensorRT test engine requires batch_size=12")

    feature_engine, post_engine = resolve_engine_paths(engine_dir)
    root = Path(ffs_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from core.foundation_stereo import TrtRunner
        from omegaconf import OmegaConf
        import yaml
    except ImportError as exc:
        raise ImportError(
            "FFS TensorRT runtime requires the FFS source tree, TensorRT, "
            "and OmegaConf dependencies"
        ) from exc

    config_path = Path(engine_dir).expanduser() / "onnx.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"FFS TensorRT config not found: {config_path}")
    with config_path.open() as handle:
        args = OmegaConf.create(yaml.safe_load(handle))
    image_size = args.get("image_size")
    if not image_size or len(image_size) != 2:
        raise ValueError(f"Invalid FFS TensorRT image_size in {config_path}")
    input_height, input_width = map(int, image_size)
    runner = TrtRunner(args, str(feature_engine), str(post_engine))
    post_batch_size = engine_batch_limit(runner.post_engine, "features_left_04")
    return FFSTensorRTModel(
        runner=runner,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        output_height=input_height,
        output_width=input_width,
        post_batch_size=post_batch_size,
    )


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    _, _, width = disp.shape
    x = torch.arange(width, device=disp.device).view(1, 1, width)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < width - 1)).float()


@torch.no_grad()
def run_stereo_matching_batch(
    model: FFSTensorRTModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if left.ndim != 4 or right.ndim != 4:
        raise ValueError("FFS TensorRT inputs must have shape (B,3,H,W)")
    batch = left.shape[0]
    original_height, original_width = left.shape[-2:]
    left, right = pad_static_batch(left.to(device), right.to(device), model.batch_size)
    left = pad_static_spatial(left, model.input_height, model.input_width).contiguous()
    right = pad_static_spatial(right, model.input_height, model.input_width).contiguous()
    _synchronize(device)
    started = time.perf_counter()
    feature_output = model.runner.run_trt(
        model.runner.feature_engine,
        model.runner.feature_context,
        {"left": left.float(), "right": right.float()},
    )
    from core.foundation_stereo import build_gwc_volume_triton

    post_inputs = {
        name: feature_output[name]
        for name in POST_FEATURE_NAMES
    }
    post_inputs["gwc_volume"] = build_gwc_volume_triton(
        feature_output["features_left_04"].half(),
        feature_output["features_right_04"].half(),
        model.runner.args.max_disp // 4,
        model.runner.cv_group,
        normalize=model.runner.args.normalize,
    )
    disp = run_post_in_chunks(
        model.runner,
        model.runner.post_engine,
        model.runner.post_context,
        post_inputs,
        model.post_batch_size,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if disp.ndim == 4 and disp.shape[1] == 1:
        disp = disp[:, 0]
    if disp.ndim != 3:
        raise ValueError(f"Unexpected FFS TensorRT disparity shape: {tuple(disp.shape)}")
    disp = disp[:batch, :original_height, :original_width].float().cpu()
    occ = _visibility(disp)
    conf = torch.ones_like(disp)
    return disp, occ, conf, elapsed


@torch.no_grad()
def run_stereo_matching(
    model: FFSTensorRTModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    disp, occ, conf, elapsed = run_stereo_matching_batch(
        model, left, right, device, **kwargs
    )
    return disp[0], occ[0], conf[0], elapsed
