"""Canonical locations for ignored model-weight artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_depth_anything_checkpoint(model_name: str, module_root: Path) -> Path | None:
    """Return the explicit or repository-level Depth Anything V2 checkpoint."""
    checkpoint_name = f"depth_anything_v2_{model_name}.pth"
    env_dir = os.environ.get("DAV2_CKPT_DIR")
    if env_dir:
        explicit = Path(env_dir).expanduser() / checkpoint_name
        if explicit.is_file():
            return explicit

    canonical = module_root.parent / "weights" / "depth-anything-ckpts" / checkpoint_name
    return canonical if canonical.is_file() else None
