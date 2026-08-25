# Left Small Hole Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, RGB-gated small-hole fill for left-view depth visualization.

**Architecture:** A standalone NumPy/OpenCV helper selects only bounded interior invalid components and fills them from compatible valid boundary depths. The depth-video driver applies it after left disparity becomes metric depth and writes timing/counts to JSON artifacts.

**Tech Stack:** Python, NumPy, OpenCV connected components, unittest.

---

### Task 1: Define conservative fill behavior with failing tests

**Files:**
- Create: `tests/test_left_hole_fill.py`

- [ ] Write tests that a 3x3 interior hole on a constant-color 2 m plane is filled, an oversized or border-touching hole remains invalid, and a color-incompatible boundary remains invalid.
- [ ] Run `./.venv/bin/python -m unittest tests.test_left_hole_fill -v` and verify it fails because `left_hole_fill` does not exist.

### Task 2: Implement the isolated fill helper

**Files:**
- Create: `stereo_center/stereo_center/left_hole_fill.py`
- Test: `tests/test_left_hole_fill.py`

- [ ] Implement `fill_small_left_holes(depth, valid, guide_bgr, max_area, color_tol)` using `connectedComponentsWithStats`, a 3x3 dilation ring, grayscale compatibility, and boundary-depth median.
- [ ] Return copied depth/valid arrays plus `filled_components` and `filled_pixels`.
- [ ] Re-run the targeted tests and verify they pass.

### Task 3: Add opt-in driver integration and metrics

**Files:**
- Modify: `stereo_center/scripts/run_depth_video.py`
- Modify: `tests/test_depth_video_timing.py`

- [ ] Add `--left-hole-fill`, `--left-hole-fill-max-area`, and `--left-hole-fill-color-tol` with conservative defaults.
- [ ] Invoke the helper only in the per-frame left-view path and add timing/count fields to `stats.json` and backend timing JSON.
- [ ] Add parser/default tests and run `./.venv/bin/python -m unittest tests.test_depth_video_timing -v`.

### Task 4: Verify visually on an eight-frame SGBM sample

**Files:**
- Output: `outputs/smoke_opencv_sgbm_left_hole_fill_12-30/`

- [ ] Run the full test suite.
- [ ] Run 8 frames with `--left-hole-fill 1`, compare valid-pixel coverage and extracted representative frames against the unfilled SGBM smoke output.
- [ ] Report filled-pixel count and state whether the effect is material enough for a full server re-run.

