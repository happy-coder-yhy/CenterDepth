"""Stereo center-depth pipeline package.

Submodules are imported lazily by callers so optional stereo backends do not
force unrelated third-party dependencies at package import time.
"""

__all__ = ["calib", "s2m2_inference", "softsplat"]
