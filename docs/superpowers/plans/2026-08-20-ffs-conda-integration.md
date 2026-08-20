# Fast Foundation Stereo Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fast Foundation Stereo as a Conda/Python stereo backend and run the prior 60s-120s benchmark segment.

**Architecture:** Implement a small FFS wrapper that matches the existing backend interface and reuse the current video pipeline. Keep Docker as fallback only if Conda dependency resolution fails on the remote server.

**Tech Stack:** Python, PyTorch, OpenCV, existing `stereo_center` pipeline, Fast-FoundationStereo source tree.

---

### Task 1: Backend Registration Tests

**Files:**
- Create: `tests/test_ffs_backend.py`
- Modify: `stereo_center/stereo_center/stereo_backend.py`
- Create: `stereo_center/stereo_center/ffs_inference.py`

- [ ] **Step 1: Write failing tests**

Create tests that assert `ffs` is registered and that checkpoint path resolution accepts both a directory and a file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=stereo_center python3 -m unittest tests.test_ffs_backend -v`

Expected: failure because `ffs_inference` and `ffs` backend support are missing.

- [ ] **Step 3: Add minimal FFS wrapper and backend registration**

Implement lazy FFS root resolution, checkpoint resolution, and backend registration without importing heavy FFS dependencies at module import time.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=stereo_center python3 -m unittest tests.test_ffs_backend -v`

Expected: all tests pass.

### Task 2: Video Script Integration

**Files:**
- Modify: `stereo_center/scripts/run_depth_video.py`

- [ ] **Step 1: Add FFS to argument choices**

Add `ffs` to `--stereo-backend` and pass `ffs_root`, `max_disp`, and `valid_iters` through `stereo_backend.load`.

- [ ] **Step 2: Add CLI options**

Add `--ffs-root` and `--ffs-valid-iters`. Reuse existing `--max-disp`.

- [ ] **Step 3: Run CLI smoke**

Run: `PYTHONPATH=stereo_center python3 stereo_center/scripts/run_depth_video.py --help`

Expected: help text includes `ffs`, `--ffs-root`, and `--ffs-valid-iters`.

### Task 3: Remote Setup and Smoke

**Files:**
- No repository files unless remote dependency issues reveal a code bug.

- [ ] **Step 1: Upload weight directory**

Upload `weights/pretrain_weights/20-26-39/` to remote `~/BothEyesDepth/CenterDepth/weights/fast_foundation_stereo/20-26-39/`.

- [ ] **Step 2: Get Fast-FoundationStereo source**

Clone or update `~/BothEyesDepth/Fast-FoundationStereo` from `https://github.com/NVlabs/Fast-FoundationStereo`.

- [ ] **Step 3: Prepare Conda runtime**

Try the current `waft` environment first for import smoke. If dependencies are missing or incompatible, create/use a separate `ffs` Conda environment.

- [ ] **Step 4: Run short remote smoke**

Run 8-16 frames with `--stereo-backend ffs`, `--scale 0.5`, and the uploaded weights.

Expected: `depth_video.mp4`, `stats.json`, and backend timing output are produced.

### Task 4: Formal Remote Experiment

**Files:**
- Remote output directory under `stereo_center/outputs/`
- Local output directory under project `outputs/`

- [ ] **Step 1: Run full segment**

Run frames 1800-3600 from `~/workspace_vdego_amvio_v6_h264/vdego-c2-48b749_2026-07-29_22-02-32_30fps/output1.mp4` with calibration from the same directory.

- [ ] **Step 2: Download artifacts**

Download the result directory to local `outputs/`.

- [ ] **Step 3: Summarize timings**

Report model load, decode, rectify, FFS forward, photometric alignment, center fusion, fill, colorize, write, main loop, and end-to-end time.
