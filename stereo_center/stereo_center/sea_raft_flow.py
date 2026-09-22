"""SEA-RAFT optical-flow adapter for temporal depth experiments.

The upstream repository is intentionally kept outside this package because its
core modules use generic import names. This adapter loads one pinned source
tree only when the experiment explicitly enables SEA-RAFT.
"""

from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F


def _sea_raft_core(root: str | Path) -> Path:
    source_root = Path(root).expanduser().resolve()
    core = source_root / "core"
    required = (core / "raft.py", core / "extractor.py", core / "layer.py")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "SEA-RAFT source tree is incomplete; expected core/raft.py, "
            "core/extractor.py, and core/layer.py under "
            f"{source_root}"
        )
    return core


def _import_upstream_raft(core: Path):
    """Import upstream modules without replacing another loaded RAFT tree."""
    for name in ("raft", "corr", "update", "extractor", "layer"):
        loaded = sys.modules.get(name)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file and core not in Path(loaded_file).resolve().parents:
            raise RuntimeError(
                "SEA-RAFT cannot share this process with another top-level "
                f"module named {name!r}: {loaded_file}. Run it in a separate "
                "experiment process."
            )
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    return importlib.import_module("raft"), importlib.import_module("extractor")


def _skip_imagenet_initialization(extractor_module) -> None:
    """Avoid an unnecessary torchvision download before loading full weights."""
    original = extractor_module.ResNetFPN._init_weights

    def initialize_without_pretrained_backbone(self, args):
        self.init_weight = False
        return original(self, args)

    extractor_module.ResNetFPN._init_weights = initialize_without_pretrained_backbone


def _load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("loading .safetensors SEA-RAFT weights requires safetensors") from exc
        return load_file(str(checkpoint))

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"SEA-RAFT checkpoint is not a state dict: {checkpoint}")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _validate_missing_shared_batchnorm_keys(
    missing: list[str], state: dict[str, torch.Tensor]
) -> None:
    """Allow only safetensors omissions caused by shared upstream BN modules."""
    for key in missing:
        marker = ".downsample.1."
        if marker not in key:
            raise RuntimeError(f"SEA-RAFT checkpoint is missing parameter: {key}")
        alias = key.replace(marker, ".bn3.")
        if alias not in state:
            raise RuntimeError(
                "SEA-RAFT checkpoint is missing a non-aliased downsample batch "
                f"normalization parameter: {key}"
            )


def load_sea_raft(
    checkpoint: str | Path,
    source_root: str | Path,
    config_path: str | Path,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load a fully specified SEA-RAFT checkpoint in evaluation mode."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SEA-RAFT checkpoint not found: {checkpoint_path}")
    if not config.is_file():
        raise FileNotFoundError(f"SEA-RAFT config not found: {config}")

    core = _sea_raft_core(source_root)
    raft_module, extractor_module = _import_upstream_raft(core)
    _skip_imagenet_initialization(extractor_module)
    args = Namespace(**json.loads(config.read_text()))
    model = raft_module.RAFT(args)
    state = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"SEA-RAFT checkpoint has unexpected keys: {unexpected}")
    _validate_missing_shared_batchnorm_keys(missing, state)
    model.eval().to(device)
    return model


def _resized_shape(height: int, width: int, input_scale: float) -> tuple[int, int]:
    if not 0.0 < float(input_scale) <= 1.0:
        raise ValueError("SEA-RAFT input_scale must be in (0, 1]")
    resized_height = max(1, round(height * float(input_scale)))
    resized_width = max(1, round(width * float(input_scale)))
    return resized_height, resized_width


@torch.no_grad()
def flow_between(
    model: torch.nn.Module,
    image0: torch.Tensor,
    image1: torch.Tensor,
    *,
    iters: int = 4,
    input_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image0-to-image1 flow and info in image0 pixel coordinates."""
    if image0.ndim != 4 or image0.shape[1] != 3:
        raise ValueError("SEA-RAFT inputs must have shape (B, 3, H, W)")
    if image1.shape != image0.shape:
        raise ValueError("SEA-RAFT image pairs must have identical shapes")
    if iters < 0:
        raise ValueError("SEA-RAFT iters must be non-negative")

    height, width = image0.shape[-2:]
    resized_height, resized_width = _resized_shape(height, width, input_scale)
    if (resized_height, resized_width) != (height, width):
        image0 = F.interpolate(
            image0, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )
        image1 = F.interpolate(
            image1, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

    output = model(image0.contiguous(), image1.contiguous(), iters=iters, test_mode=True)
    flow = output["final"].float()
    info = output["info"][-1].float()
    if not torch.isfinite(flow).all() or not torch.isfinite(info).all():
        raise RuntimeError("SEA-RAFT returned non-finite flow or info")

    if (resized_height, resized_width) != (height, width):
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
        flow[:, 0] *= width / resized_width
        flow[:, 1] *= height / resized_height
        info = F.interpolate(info, size=(height, width), mode="area")
    return flow, info
