# OpenCV StereoSGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently configurable OpenCV StereoSGBM left-view stereo backend to the existing depth-video pipeline.

**Architecture:** A new parameter-only inference module creates a CPU `cv2.StereoSGBM` matcher per frame and returns disparity, visibility and unit confidence using the existing backend contract. The existing driver registers the backend, parses its parameters, blocks center/bidirectional modes, and persists reproducible SGBM metadata in backend-specific timing and stats artifacts.

**Tech Stack:** Python 3, OpenCV `cv2.StereoSGBM`, NumPy, PyTorch tensors, `unittest`.

---

### Task 1: Define the SGBM backend contract with failing tests

**Files:**
- Create: `tests/test_opencv_sgbm_backend.py`
- Modify: `tests/test_depth_video_timing.py`

- [ ] **Step 1: Write failing registry, validation and synthetic-disparity tests**

```python
def test_backend_registry_includes_opencv_sgbm(self):
    self.assertIn("opencv_sgbm", stereo_backend.BACKENDS)

def test_load_rejects_invalid_sgbm_parameters(self):
    with self.assertRaisesRegex(ValueError, "multiple of 16"):
        stereo_backend.load("opencv_sgbm", "ignored", "", "cpu", sgbm_num_disparities=20)

def test_recovers_known_horizontal_disparity(self):
    # Build a random RGB image and a right image shifted by eight pixels.
    # Assert the median valid interior disparity is 8 +/- 1.
```

Add driver tests for `opencv_sgbm_timing.json`, `--sgbm-*` defaults and rejection of `center` or `bi=1`.

- [ ] **Step 2: Run the new tests to verify the expected missing-backend failure**

Run: `./.venv/bin/python -m unittest tests.test_opencv_sgbm_backend tests.test_depth_video_timing -v`

Expected: FAIL because `opencv_sgbm` is absent from `BACKENDS` and SGBM helper functions are undefined.

### Task 2: Implement the parameter-only SGBM inference module

**Files:**
- Create: `stereo_center/stereo_center/opencv_sgbm_inference.py`
- Test: `tests/test_opencv_sgbm_backend.py`

- [ ] **Step 1: Add the model data structure and loader**

```python
@dataclass(frozen=True)
class OpenCVSGBMModel:
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    p1: int | None = None
    p2: int | None = None
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY
```

Validate the positive multiple-of-16 disparity range, odd block size from 1 to 255, non-negative filter values, and `p2 > p1` when both are explicit. Derive omitted `p1` and `p2` as `8*3*block_size**2` and `32*3*block_size**2`.

- [ ] **Step 2: Add matcher construction and the backend `run_stereo_matching` function**

```python
raw = matcher.compute(_gray_uint8(left_image), _gray_uint8(right_image))
disparities.append(torch.from_numpy(raw.astype(np.float32) / 16.0))
```

Reuse BM's RGB-to-grayscale conversion and left-reference visibility definition. Require matching `(B, 3, H, W)` inputs, process every batch item on CPU, return `(disp, occ, conf, elapsed)`, and create a matcher per item so no mutable OpenCV state crosses calls.

- [ ] **Step 3: Run the backend tests**

Run: `./.venv/bin/python -m unittest tests.test_opencv_sgbm_backend -v`

Expected: PASS, including the synthetic 8-pixel disparity recovery.

### Task 3: Register the backend and expose reproducible CLI metadata

**Files:**
- Modify: `stereo_center/stereo_center/stereo_backend.py`
- Modify: `stereo_center/scripts/run_depth_video.py`
- Modify: `tests/test_depth_video_timing.py`

- [ ] **Step 1: Add dispatch registration**

```python
BACKENDS = ("s2m2", "waft", "las2", "ffs", "opencv_bm", "opencv_sgbm")
```

Route `get_backend`, `load`, and `run` for `opencv_sgbm` to `opencv_sgbm_inference`; do not add it to bidirectional dispatch.

- [ ] **Step 2: Add CLI and metadata helpers**

```python
parser.add_argument("--sgbm-num-disparities", type=int, default=128)
parser.add_argument("--sgbm-block-size", type=int, default=5)
parser.add_argument("--sgbm-p1", type=int, default=None)
parser.add_argument("--sgbm-p2", type=int, default=None)
parser.add_argument("--sgbm-mode", choices=("sgbm", "hh", "3way", "hh4"), default="3way")
```

Add the remaining SGBM filter flags, map text modes to OpenCV constants in the backend loader, set no-weights resolution for SGBM, reject any mode except left with `bi=0`, pass `sgbm_parameters(args)` to model load and serialize it in timing and stats under `sgbm_parameters`.

- [ ] **Step 3: Run driver tests**

Run: `./.venv/bin/python -m unittest tests.test_depth_video_timing -v`

Expected: PASS and no existing timing test regressions.

### Task 4: Run regression tests and a local smoke video

**Files:**
- Test: `tests/`
- Output: `outputs/smoke_opencv_sgbm_local_12-30/`

- [ ] **Step 1: Run all tests**

Run: `./.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Process eight frames with SGBM**

Run:

```bash
cd stereo_center
../.venv/bin/python scripts/run_depth_video.py \
  --video ../dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/vdego-c2-48b749_2026-07-28_12-30-24_30fps/output.mp4 \
  --calib ../dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/vdego-c2-48b749_2026-07-28_12-30-24_30fps/calibration.json \
  --stereo-backend opencv_sgbm --output-view left --bi 0 --scale 0.5 \
  --batch-size 8 --end-frame 8 --outdir ../outputs/smoke_opencv_sgbm_local_12-30
```

Expected: playable `depth_video.mp4`, valid `opencv_sgbm_timing.json`, and zero values for center-only timing stages.

### Task 5: Commit the tested implementation

**Files:**
- Create: `stereo_center/stereo_center/opencv_sgbm_inference.py`
- Modify: `stereo_center/stereo_center/stereo_backend.py`
- Modify: `stereo_center/scripts/run_depth_video.py`
- Create: `tests/test_opencv_sgbm_backend.py`
- Modify: `tests/test_depth_video_timing.py`
- Create: `docs/superpowers/specs/2026-08-25-opencv-sgbm-design.md`
- Create: `docs/superpowers/plans/2026-08-25-opencv-sgbm.md`

- [ ] **Step 1: Inspect final diff and commit**

Run: `git diff --check && git status --short && git add stereo_center/stereo_center/opencv_sgbm_inference.py stereo_center/stereo_center/stereo_backend.py stereo_center/scripts/run_depth_video.py tests/test_opencv_sgbm_backend.py tests/test_depth_video_timing.py docs/superpowers/specs/2026-08-25-opencv-sgbm-design.md docs/superpowers/plans/2026-08-25-opencv-sgbm.md && git commit -m "feat: add OpenCV StereoSGBM backend"`

Expected: a commit containing only SGBM backend integration, tests, and implementation documentation.

