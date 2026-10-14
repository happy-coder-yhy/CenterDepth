"""Compatibility imports for existing depth Zarr readers.

The standalone zpds-depth-cp package owns both current and legacy codecs.
"""

from zpds_depth_cp.codecs import DepthLogSZ3, DepthSZ3

__all__ = ["DepthLogSZ3", "DepthSZ3"]
