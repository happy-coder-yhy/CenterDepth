#!/usr/bin/env python3
"""Build fixed-shape FP16 TensorRT engines for the FFS pipeline.

The engines use the upstream FFS split: feature extraction and post-processing
run in TensorRT, while the GWC volume between them remains Triton at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf


def export_trace_batch_size(component: str, batch_size: int) -> int:
    """Keep FFS feature slicing static at the batch expected by its engine."""
    if component not in ("feature", "post", "both"):
        raise ValueError(f"Unknown FFS TensorRT export component: {component}")
    if batch_size < 1:
        raise ValueError("FFS TensorRT export batch size must be positive")
    return batch_size if component == "feature" else 1


def export_trace_batch_sizes(component: str, batch_size: int) -> tuple[int, int]:
    """Return independent static batches for the two split FFS exports."""
    export_trace_batch_size(component, batch_size)
    return (
        export_trace_batch_size("feature", batch_size),
        export_trace_batch_size("post", batch_size),
    )


def validate_engine_dimensions(height: int, width: int) -> tuple[int, int]:
    """Validate dimensions required by FFS's four-stage feature pyramid."""
    height, width = int(height), int(width)
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError(
            f"FFS TensorRT engine dimensions must be positive multiples of 32, got {height}x{width}"
        )
    return height, width


def network_creation_flags(trt: object) -> int:
    """Use explicit-batch flags only on TensorRT versions that expose them."""
    creation_flag = getattr(
        getattr(trt, "NetworkDefinitionCreationFlag", object()),
        "EXPLICIT_BATCH",
        None,
    )
    return 0 if creation_flag is None else 1 << int(creation_flag)


def enable_fp16(trt: object, config: object) -> bool:
    """Enable the legacy FP16 builder flag when the installed API exposes it."""
    fp16_flag = getattr(getattr(trt, "BuilderFlag", object()), "FP16", None)
    if fp16_flag is None:
        return False
    config.set_flag(fp16_flag)
    return True


def needs_fp16_onnx_export(trt: object) -> bool:
    """Use typed FP16 ONNX only when TensorRT lacks its FP16 builder flag."""
    return not hasattr(getattr(trt, "BuilderFlag", object()), "FP16")


def prepare_model_for_export(model: object, fp16: bool) -> torch.dtype:
    """Use FP16 ONNX tensors when TensorRT cannot enable FP16 at build time."""
    if not fp16:
        return torch.float32
    model.half()
    model.dtype = torch.float16
    return torch.float16


def export_onnx(
    model_path: Path,
    ffs_root: Path,
    output_dir: Path,
    height: int,
    width: int,
    batch_size: int,
    valid_iters: int,
    max_disp: int,
    component: str,
    fp16: bool,
) -> tuple[Path, Path]:
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    sys.path.insert(0, str(ffs_root))
    from core.foundation_stereo import (  # type: ignore
        TrtFeatureRunner,
        TrtPostRunner,
        build_gwc_volume_triton,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.args.max_disp = int(max_disp)
    model.args.valid_iters = int(valid_iters)
    model.args.mixed_precision = False
    model.cuda().eval()
    tensor_dtype = prepare_model_for_export(model, fp16)
    feature_runner = TrtFeatureRunner(model).cuda().eval()
    post_runner = TrtPostRunner(model).cuda().eval() if component != "feature" else None
    # The feature runner slices concatenated left/right features using B. ONNX
    # tracing records that slice size, so a batch-12 feature engine must trace
    # with 12 samples. TensorRT 11.2 cannot build this post graph at batch 12,
    # so its supported engine remains batch 1 and runs in chunks at runtime.
    feature_trace_batch, post_trace_batch = export_trace_batch_sizes(
        component, batch_size
    )
    left = torch.randn(
        feature_trace_batch, 3, height, width, device="cuda", dtype=tensor_dtype
    ) * 255
    right = torch.randn(
        feature_trace_batch, 3, height, width, device="cuda", dtype=tensor_dtype
    ) * 255

    feature_path = output_dir / "feature_runner.onnx"
    post_path = output_dir / "post_runner.onnx"
    if component in ("feature", "both"):
        torch.onnx.export(
            feature_runner,
            (left, right),
            feature_path,
            opset_version=17,
            input_names=["left", "right"],
            output_names=[
                "features_left_04", "features_left_08", "features_left_16",
                "features_left_32", "features_right_04", "stem_2x",
            ],
            do_constant_folding=True,
            dynamo=False,
        )
    if component in ("post", "both"):
        assert post_runner is not None
        with torch.no_grad():
            post_left = left[:post_trace_batch]
            post_right = right[:post_trace_batch]
            features = feature_runner(post_left, post_right)
            gwc = build_gwc_volume_triton(
                features[0].half(), features[4].half(),
                max_disp // 4, model.cv_group,
                normalize=model.args.normalize,
            )
            post_inputs = tuple(x.to(tensor_dtype) for x in (*features, gwc))
            post_runner(*post_inputs)
        torch.onnx.export(
            post_runner,
            post_inputs,
            post_path,
            opset_version=17,
            input_names=[
                "features_left_04", "features_left_08", "features_left_16",
                "features_left_32", "features_right_04", "stem_2x", "gwc_volume",
            ],
            output_names=["disp"],
            do_constant_folding=True,
            dynamo=False,
        )
    with (output_dir / "onnx.yaml").open("w") as handle:
        config = OmegaConf.to_container(model.args, resolve=True)
        config["image_size"] = [height, width]
        yaml.safe_dump(config, handle)
    return feature_path, post_path


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: int,
    batch_size: int,
    fp16: bool,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        network_creation_flags(trt)
    )
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib) << 30
    )
    if fp16:
        enable_fp16(trt, config)
    network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
    dynamic_inputs = [
        input_tensor for input_tensor in network_inputs if -1 in tuple(input_tensor.shape)
    ]
    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for input_tensor in dynamic_inputs:
            shape = list(input_tensor.shape)
            min_shape = list(shape)
            opt_shape = list(shape)
            max_shape = list(shape)
            min_shape[0] = 1
            opt_shape[0] = batch_size
            max_shape[0] = batch_size
            profile.set_shape(
                input_tensor.name,
                tuple(min_shape), tuple(opt_shape), tuple(max_shape),
            )
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path}")
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ffs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument(
        "--only", choices=["feature", "post", "both"], default="both",
        help="Build only one split engine when validating a specific batch profile.",
    )
    parser.add_argument(
        "--post-fp32", action="store_true",
        help="Build the post runner in FP32 when its FP16 tactics are unavailable.",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Reuse existing feature_runner.onnx and post_runner.onnx in --outdir.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("FFS TensorRT engine batch size must be positive")
    validate_engine_dimensions(args.height, args.width)
    if args.skip_export:
        feature_onnx = args.outdir / "feature_runner.onnx"
        post_onnx = args.outdir / "post_runner.onnx"
        required = []
        if args.only in ("feature", "both"):
            required.append(feature_onnx)
        if args.only in ("post", "both"):
            required.append(post_onnx)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("--skip-export missing: " + ", ".join(missing))
    else:
        import tensorrt as trt

        feature_onnx, post_onnx = export_onnx(
            args.model.expanduser().resolve(), args.ffs_root.expanduser().resolve(),
            args.outdir.expanduser().resolve(), args.height, args.width,
            args.batch_size, args.iters, args.max_disp, args.only,
            fp16=(not args.post_fp32 and needs_fp16_onnx_export(trt)),
        )
    if args.only in ("feature", "both"):
        build_engine(
            feature_onnx, args.outdir / "feature_runner.engine",
            args.workspace_gib, args.batch_size, fp16=True,
        )
    if args.only in ("post", "both"):
        build_engine(
            post_onnx, args.outdir / "post_runner.engine",
            args.workspace_gib, args.batch_size, fp16=not args.post_fp32,
        )
    print(f"[trt] engines written to {args.outdir}")


if __name__ == "__main__":
    main()
