# Fast Foundation Stereo Conda Integration Design

## Goal

Add Fast Foundation Stereo (FFS) as a stereo backend for the existing center-depth video pipeline, then run the same 60s-120s benchmark segment used for WAFT.

## Installation Choice

Use a Conda/Python integration first. The current project already runs stereo models through `stereo_backend.py`, then reuses the same rectification, center-view fusion, fill, colorization, video writing, and timing code. A Docker-only integration would isolate dependencies well, but it would force file-based handoff between the official FFS demo and this pipeline, making batching and timing less comparable.

The remote server has Docker available, but it also has a working CUDA Python environment on the A100. If FFS dependencies conflict with the current `waft` environment, create or use a separate Conda environment for FFS instead of changing the existing environment in place.

## Architecture

Add `stereo_center/stereo_center/ffs_inference.py` with the same public shape as the existing backends:

- `load_ffs(model_type, weights_dir, device, ffs_root, max_disp, valid_iters)`
- `run_stereo_matching(...)`
- `run_stereo_matching_bi(...)`
- `run_stereo_matching_bi_batch(...)`

Register `ffs` in `stereo_center/stereo_center/stereo_backend.py`. Extend `stereo_center/scripts/run_depth_video.py` so `--stereo-backend ffs` is accepted and the existing non-WAFT batch path calls `stereo_backend.run_bi_batch`.

## Data Flow

The pipeline stays unchanged around the model:

1. Decode frames from the stereo video.
2. Split left/right images.
3. Rectify and scale with existing calibration code.
4. Convert rectified BGR images to RGB tensors.
5. Run FFS to produce left/right disparity.
6. Generate visibility or left-right consistency masks.
7. Reuse existing photometric alignment, center fusion, disocclusion fill, depth colorization, and MP4 writing.

## FFS Model Interface

The official demo loads a serialized model checkpoint directly with `torch.load`, sets `model.args.valid_iters` and `model.args.max_disp`, pads inputs to a multiple of 32 with FFS `InputPadder`, and calls `model.forward` with `test_mode=True`. The wrapper should follow that pattern and only add minimal batching and timing around it.

The local provided weight directory contains:

- `weights/pretrain_weights/20-26-39/model_best_bp2_serialize.pth`
- `weights/pretrain_weights/20-26-39/cfg.yaml`

The wrapper resolves either a directory containing `model_best_bp2_serialize.pth` or a direct checkpoint path.

## Error Handling

If the FFS source tree is missing, raise a clear error asking for `--ffs-root` or `FFS_ROOT`.

If the checkpoint is missing, raise a clear error showing the checked path.

If FFS dependencies cannot import locally, the local smoke test should still verify path resolution and backend registration. Full CUDA smoke and formal experiment happen on the remote server.

## Testing

Add unit tests for:

- FFS checkpoint path resolution from a directory.
- FFS checkpoint path resolution from a direct file.
- `stereo_backend.BACKENDS` includes `ffs`.
- `stereo_backend.get_backend("ffs")` imports the FFS wrapper when dependencies are not required at import time.

Manual verification:

- Local smoke test with Python import and tests.
- Remote import smoke after cloning/installing FFS.
- Remote short video smoke before the full 60s-120s run.
- Full 60s-120s run with timing artifacts and downloaded outputs.
