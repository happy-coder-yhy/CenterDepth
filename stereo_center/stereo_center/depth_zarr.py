"""Compatibility imports for the standalone S1 depth storage package."""

from zpds_depth_cp.storage import (
    DEPTH_CHUNK_FRAMES,
    DEPTH_STORAGE_ALPHA,
    DepthZarrWriter,
)

__all__ = ["DEPTH_CHUNK_FRAMES", "DEPTH_STORAGE_ALPHA", "DepthZarrWriter"]
