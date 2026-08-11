"""立体匹配后端统一入口（s2m2 | waft）。

waft 相关模块按需惰性导入，避免 s2m2 路径依赖 peft/timm/yacs
（服务器 allbase_env 可继续独立运行 s2m2）。
"""

from __future__ import annotations

BACKENDS = ("s2m2", "waft")


def get_backend(name: str):
    """返回后端模块（s2m2_inference / waft_inference）。"""
    if name == "s2m2":
        from . import s2m2_inference

        return s2m2_inference
    if name == "waft":
        from . import waft_inference

        return waft_inference
    raise ValueError(f"未知立体匹配后端: {name}（可选: {', '.join(BACKENDS)}）")


def load(backend: str, model_type: str, weights_dir: str, device: str, **kwargs):
    """按后端加载模型；s2m2 额外支持 num_refine。"""
    if backend == "s2m2":
        from . import s2m2_inference

        return s2m2_inference.load_s2m2(
            model_type, weights_dir, kwargs.get("num_refine", 3), device
        )
    if backend == "waft":
        from . import waft_inference

        return waft_inference.load_waft(model_type, weights_dir, device)
    raise ValueError(f"未知立体匹配后端: {backend}")


def run(backend: str, model, left, right, device: str, **kwargs):
    """按后端调用推理，返回 (disp, occ, conf, elapsed)。"""
    if backend == "s2m2":
        from . import s2m2_inference

        return s2m2_inference.run_stereo_matching(
            model, left, right, device, use_amp=kwargs.get("use_amp", False)
        )
    if backend == "waft":
        from . import waft_inference

        return waft_inference.run_stereo_matching(model, left, right, device, **kwargs)
    raise ValueError(f"未知立体匹配后端: {backend}")
