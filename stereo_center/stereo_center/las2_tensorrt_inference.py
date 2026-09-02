"""TensorRT runtime for the ONNX-compatible LAS2 model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ENGINE_NAME = "las2.engine"


@dataclass
class LAS2TensorRTModel:
    runner: object
    batch_size: int
    input_height: int
    input_width: int
    max_disp: int
    runtime: str = "tensorrt"


def resolve_engine_path(engine_dir: str | Path) -> Path:
    root = Path(engine_dir).expanduser()
    path = root if root.is_file() else root / ENGINE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"LAS2 TensorRT engine not found: {path}")
    return path


def engine_batch_limit(engine: object, input_name: str = "left") -> int:
    """Return the maximum batch from TensorRT optimization profile zero."""
    _, _, maximum = engine.get_tensor_profile_shape(input_name, 0)
    limit = int(maximum[0])
    if limit < 1:
        raise ValueError(f"Invalid LAS2 TensorRT batch profile: {maximum}")
    return limit


def pad_static_spatial(image: torch.Tensor, input_height: int, input_width: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    if height > input_height or width > input_width:
        raise ValueError(
            f"LAS2 TensorRT engine expects at most {input_height}x{input_width}, "
            f"got {height}x{width}"
        )
    return F.pad(image, (0, input_width - width, 0, input_height - height))


def run_engine_in_chunks(runner: object, inputs: dict[str, torch.Tensor], batch_limit: int) -> torch.Tensor:
    if batch_limit < 1:
        raise ValueError("LAS2 TensorRT batch limit must be positive")
    batch = next(iter(inputs.values())).shape[0]
    if batch < 1 or any(t.shape[0] != batch for t in inputs.values()):
        raise ValueError("LAS2 TensorRT inputs must share a positive batch size")
    outputs = []
    for start in range(0, batch, batch_limit):
        stop = min(start + batch_limit, batch)
        output = runner.run_trt({name: value[start:stop] for name, value in inputs.items()})
        if "disparity" not in output:
            raise ValueError("LAS2 TensorRT engine did not return 'disparity'")
        outputs.append(output["disparity"])
    return torch.cat(outputs, dim=0)


class _TensorRTRunner:
    def __init__(self, engine_path: Path):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()

    @staticmethod
    def _dtype(dtype, trt):
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")
        return mapping[dtype]

    def _io_names(self, mode):
        return [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == mode
        ]

    def run_trt(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, tensor in list(inputs.items()):
            expected = self._dtype(self.engine.get_tensor_dtype(name), self.trt)
            tensor = tensor.to(expected) if tensor.dtype != expected else tensor
            inputs[name] = tensor.contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
        outputs = {}
        for name in self._io_names(self.trt.TensorIOMode.OUTPUT):
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype(self.engine.get_tensor_dtype(name), self.trt)
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("LAS2 TensorRT execution failed")
        return outputs


def load_las2_tensorrt(
    engine_dir: str | Path,
    device: str = "cuda",
    batch_size: int = 16,
    max_disp: int = 192,
) -> LAS2TensorRTModel:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("LAS2 TensorRT runtime requires a CUDA device")
    runner = _TensorRTRunner(resolve_engine_path(engine_dir))
    limit = engine_batch_limit(runner.engine)
    if batch_size > limit:
        raise ValueError(f"Requested LAS2 batch_size={batch_size}, engine limit is {limit}")
    return LAS2TensorRTModel(runner, limit, 672, 800, int(max_disp))


def _visibility(disp: torch.Tensor) -> torch.Tensor:
    _, height, width = disp.shape
    x = torch.arange(width, device=disp.device).view(1, 1, width)
    return ((disp >= 0.5) & (x - disp >= 0) & (disp < width - 1)).float()


@torch.no_grad()
def run_stereo_matching_batch(
    model: LAS2TensorRTModel,
    left: torch.Tensor,
    right: torch.Tensor,
    device: str = "cuda",
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if left.ndim != 4 or right.ndim != 4 or left.shape[0] != right.shape[0]:
        raise ValueError("LAS2 TensorRT inputs must be matching (B,3,H,W) tensors")
    batch, height, width = left.shape[0], left.shape[-2], left.shape[-1]
    left = pad_static_spatial(left.to(device), model.input_height, model.input_width)
    right = pad_static_spatial(right.to(device), model.input_height, model.input_width)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    disp = run_engine_in_chunks(
        model.runner,
        {"left": left, "right": right},
        model.batch_size,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if disp.ndim == 4 and disp.shape[1] == 1:
        disp = disp[:, 0]
    if disp.ndim != 3:
        raise ValueError(f"Unexpected LAS2 TensorRT output shape: {tuple(disp.shape)}")
    disp = disp[:batch, :height, :width].float().cpu()
    return disp, _visibility(disp), torch.ones_like(disp), elapsed


@torch.no_grad()
def run_stereo_matching(model, left, right, device="cuda", **kwargs):
    disp, occ, conf, elapsed = run_stereo_matching_batch(model, left, right, device, **kwargs)
    if disp.shape[0] == 1:
        return disp[0], occ[0], conf[0], elapsed
    return disp, occ, conf, elapsed
