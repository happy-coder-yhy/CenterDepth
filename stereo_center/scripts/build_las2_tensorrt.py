#!/usr/bin/env python3
"""Export LAS2-L to ONNX and build a dynamic-batch TensorRT engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def export_trace_batch_size(max_batch: int) -> int:
    """Trace one sample; LAS2's batch axis remains dynamic in the ONNX graph."""
    if max_batch < 1:
        raise ValueError("LAS2 TensorRT max batch must be positive")
    return 1


def _patch_onnx_unsupported_ops(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    import core.liteanystereov2 as liteanystereov2
    import core.submodule as submodule
    import torch.nn.functional as F

    def context_upsample(depth_low, up_weights):
        b, c, h, w = depth_low.shape
        eye = torch.eye(9, dtype=depth_low.dtype, device=depth_low.device).reshape(9, 1, 3, 3)
        unfolded = F.conv2d(depth_low.reshape(b, c, h, w), eye, padding=1)
        unfolded = F.interpolate(unfolded.reshape(b, -1, h, w), (h * 4, w * 4), mode="nearest")
        return torch.sum(unfolded.reshape(b, 9, h * 4, w * 4) * up_weights, dim=1, keepdim=True)

    def build_correlation_volume(left_feature, right_feature, max_disp):
        batch, channels, height, width = left_feature.shape
        left_volume = left_feature.unsqueeze(2).expand(batch, channels, max_disp, height, width)
        padded = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
        shifted = torch.stack([padded[:, :, :, index:index + width] for index in range(max_disp)], dim=3)
        right_volume = torch.flip(shifted, [3]).permute(0, 1, 3, 2, 4)
        return (left_volume * right_volume).mean(dim=1).contiguous()

    def build_gwc_volume(left_feature, right_feature, max_disp, num_groups):
        batch, channels, height, width = left_feature.shape
        channels_per_group = channels // num_groups
        left_volume = left_feature.unsqueeze(2).expand(batch, channels, max_disp, height, width)
        padded = F.pad(right_feature, (max_disp - 1, 0, 0, 0))
        shifted = torch.stack([padded[:, :, :, index:index + width] for index in range(max_disp)], dim=3)
        target = torch.flip(shifted, [3]).permute(0, 1, 3, 2, 4)
        left_volume = left_volume.view(batch, num_groups, channels_per_group, max_disp, height, width)
        target = target.view(batch, num_groups, channels_per_group, max_disp, height, width)
        return (left_volume * target).mean(dim=2).contiguous()

    submodule.context_upsample = context_upsample
    liteanystereov2.context_upsample = context_upsample
    liteanystereov2.build_correlation_volume = build_correlation_volume
    submodule.build_gwc_volume_fast = build_gwc_volume
    liteanystereov2.build_gwc_volume_fast = build_gwc_volume


def export_onnx(source_root: Path, checkpoint: Path, output: Path, height: int, width: int, batch: int, max_disp: int) -> None:
    _patch_onnx_unsupported_ops(source_root)
    from core.models import build_model, load_model_weights
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, left, right):
            return self.model(left, right, max_disp=max_disp, test_mode=True)

    model = build_model("las2", model_size="l", max_disp=max_disp)
    load_model_weights(model, torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    model.eval().cuda()
    wrapper = Wrapper(model).eval().cuda()
    trace_batch = export_trace_batch_size(batch)
    left = torch.randint(0, 256, (trace_batch, 3, height, width), dtype=torch.float32, device="cuda")
    right = torch.randint(0, 256, (trace_batch, 3, height, width), dtype=torch.float32, device="cuda")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (left, right),
        output,
        input_names=["left", "right"],
        output_names=["disparity"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={"left": {0: "batch"}, "right": {0: "batch"}, "disparity": {0: "batch"}},
    )


def build_engine(onnx_path: Path, engine_path: Path, max_batch: int, workspace_gib: int) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = list(tensor.shape)
        shape[0] = 1
        minimum = tuple(shape)
        shape[0] = max_batch
        maximum = tuple(shape)
        profile.set_shape(tensor.name, minimum, maximum, maximum)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build LAS2 engine")
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--workspace-gib", type=int, default=8)
    args = parser.parse_args()
    onnx = args.outdir / "las2.onnx"
    engine = args.outdir / "las2.engine"
    export_onnx(args.source_root.expanduser().resolve(), args.checkpoint.expanduser().resolve(), onnx, args.height, args.width, args.max_batch, args.max_disp)
    build_engine(onnx, engine, args.max_batch, args.workspace_gib)
    (args.outdir / "engine.yaml").write_text(
        f"image_size: [{args.height}, {args.width}]\nmax_disp: {args.max_disp}\nmax_batch: {args.max_batch}\n",
        encoding="utf-8",
    )
    print(f"[trt] LAS2 engine written to {engine}")


if __name__ == "__main__":
    main()
