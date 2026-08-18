"""立体匹配后端统一入口（s2m2 | waft）。

waft 相关模块按需惰性导入，避免 s2m2 路径依赖 peft/timm/yacs
（服务器 allbase_env 可继续独立运行 s2m2）。
"""

from __future__ import annotations

BACKENDS = ("s2m2", "waft", "las2")


def get_backend(name: str):
    """返回后端模块（s2m2_inference / waft_inference）。"""
    if name == "s2m2":
        from . import s2m2_inference

        return s2m2_inference
    if name == "waft":
        from . import waft_inference

        return waft_inference
    if name == "las2":
        from . import las2_inference

        return las2_inference
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

        return waft_inference.load_waft(
            model_type, weights_dir, device, iters=kwargs.get("iters")
        )
    if backend == "las2":
        from . import las2_inference

        return las2_inference.load_las2(
            model_type, weights_dir, device,
            max_disp=kwargs.get("max_disp", 192),
            las_root=kwargs.get("las_root"),
        )
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
    if backend == "las2":
        from . import las2_inference

        return las2_inference.run_stereo_matching(model, left, right, device, **kwargs)
    raise ValueError(f"未知立体匹配后端: {backend}")


def run_bi(backend: str, model, left, right, device: str, **kwargs):
    """按后端调用双向推理，返回 (dL, dR, occL, occR, confL, confR, elapsed)。"""
    if backend == "s2m2":
        from . import s2m2_inference

        return s2m2_inference.run_stereo_matching_bi(
            model, left, right, device, use_amp=kwargs.get("use_amp", False)
        )
    if backend == "waft":
        from . import waft_inference

        return waft_inference.run_stereo_matching_bi(model, left, right, device, **kwargs)
    if backend == "las2":
        from . import las2_inference

        return las2_inference.run_stereo_matching_bi(model, left, right, device, **kwargs)
    raise ValueError(f"未知立体匹配后端: {backend}")


def run_bi_batch(backend: str, model, left, right, device: str, **kwargs):
    """按后端调用批量双向推理（深度视频用），返回 (dL,dR,occL,occR,confL,confR,elapsed)。"""
    if backend == "waft":
        from . import waft_inference

        return waft_inference.run_stereo_matching_bi_batch(model, left, right, device, **kwargs)
    if backend == "las2":
        from . import las2_inference

        return las2_inference.run_stereo_matching_bi_batch(model, left, right, device, **kwargs)
    raise ValueError(f"未知立体匹配后端: {backend}")
