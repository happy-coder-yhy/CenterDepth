# RoMa v2 独立光流评估 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入 RoMa v2 独立光流评估后端，并在 A100 上完成奥比中光 `010031` 左目完整视频推理，输出可连续播放的五路光流可视化视频、原始双向光流和性能/质量报告。

**Architecture:** 复用现有 `RoMaFlow` 的双向稠密匹配能力，以独立 CLI 解码单目视频相邻帧。第一遍推理把正反向光流写入 NPY memmap 并统计统一 P99 色标，第二遍解码视频并生成色轮、幅度、箭头、诊断和六宫格总览 MP4；整个流程不导入 FFS 或深度模块。

**Tech Stack:** Python 3、PyTorch、RoMa v2、NumPy、OpenCV、`unittest`、A100 CUDA。

**Spec:** `docs/2026-09-22-独立光流评估pipeline设计.md`

## Global Constraints

- 首阶段只实现 `roma-v2` 后端，但统一结果类型不得依赖 RoMa 内部张量名称。
- 不修改 FFS、TensorRT engine、深度视频入口或清洗切片代码。
- RoMa v2 必须离线读取 `/home/opsuser/.cache/torch/hub/checkpoints/romav2.0.1.pt`，缺失时立即失败，不允许隐式下载。
- A100 源码固定为 `/home/opsuser/BothEyesDepth/third_party/RoMaV2`，revision `95c9968145c8906b7b59383258e9f73b02853d89`。
- 真实视频只报告自一致性代理指标；AEPE、BadPix、KITTI outlier 只能用于已知真值的合成平移。
- 所有视频帧使用同一 `flow_color_max_px`，默认由全片确定性稀疏采样的 P99 得到，不允许逐帧归一化。
- 原始光流为 `float32 (N,H,W,2)`，通道为 `(u,v)`，单位为 `px / frame_stride`。
- A100 完整实验结束后，将五个 MP4、`metrics.json` 和 `manifest.json` 下载到本地桌面。

本计划是总设计的第一阶段，只交付 RoMa v2 视频输入与 `010031` 全片验证。
总设计中的静态图对、合成平移真值模式和 SEA-RAFT 后端不在本阶段实现；它们将
复用本阶段建立的结果、诊断与可视化接口，单独规划和验收。

## Review Focus

- 权重存在但路径不是标准 Torch Hub `hub/checkpoints/romav2.0.1.pt` 时，必须在模型构造前给出可操作错误，不触发网络请求；Task 1 测试覆盖。
- 视频元数据帧数大于实际可解码帧数时，缓存形状必须基于预扫描结果而不是容器声明；Task 4 测试覆盖。
- 正反向光流采样落在图像外时，诊断像素必须标为无效，不得用边界复制产生虚假低残差；Task 2 测试覆盖。
- 全零流、包含 NaN/Inf 的流以及 `vmax <= 0` 必须有确定行为；Task 3 测试覆盖。
- 视频 writer 创建失败或任一输出视频帧数不等于有效帧对数时，运行必须失败且 manifest 不标记成功；Task 5 测试覆盖。

---

## File Map

| Operation | Path | Responsibility |
| --- | --- | --- |
| Modify | `stereo_center/stereo_center/romav2_flow.py` | 显式源码根目录、Torch Hub 权重缓存预检、双向流统一结果 |
| Create | `stereo_center/stereo_center/flow_diagnostics.py` | backward warp、前后向一致性、RGB 重投影残差 |
| Create | `stereo_center/stereo_center/flow_visualization.py` | 色轮、幅度、箭头、诊断和 2x3 总览帧 |
| Create | `stereo_center/stereo_center/flow_evaluation.py` | 视频预扫描、帧对生成、memmap、计时、manifest/metrics |
| Create | `stereo_center/scripts/run_flow_evaluation.py` | 独立 CLI 和五路 MP4 编码 |
| Create | `tests/test_romav2_flow.py` | 离线加载与输出契约测试 |
| Create | `tests/test_flow_diagnostics.py` | 几何诊断测试 |
| Create | `tests/test_flow_visualization.py` | 可视化稳定性测试 |
| Create | `tests/test_flow_evaluation.py` | 假模型端到端与视频失败测试 |
| Create | `docs/2026-09-22-010031-RoMa-v2完整视频实验记录.md` | A100 命令、性能、产物和异常记录 |

### Task 1: RoMa v2 离线加载适配器

**Files:**
- Modify: `stereo_center/stereo_center/romav2_flow.py`
- Create: `tests/test_romav2_flow.py`

**Interfaces:**
- Consumes: RoMa v2 源码根目录、`romav2.0.1.pt` 路径、setting 和 `B,3,H,W` RGB 张量。
- Produces: `RoMaFlowResult(forward, backward, overlap_forward, overlap_backward)`；四个张量均恢复到输入分辨率。

- [ ] **Step 1: Write the failing offline-loader tests**

```python
class RoMaOfflineLoadTests(unittest.TestCase):
    def test_prepare_source_rejects_missing_package(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "src/romav2"):
                prepare_romav2_source(root)

    def test_prepare_checkpoint_rejects_nonstandard_cache_path(self):
        with tempfile.TemporaryDirectory() as root:
            bad = Path(root) / "romav2.0.1.pt"
            bad.touch()
            with self.assertRaisesRegex(ValueError, "hub/checkpoints"):
                prepare_romav2_checkpoint(bad)

    def test_pair_returns_named_bidirectional_result(self):
        fake = object.__new__(RoMaFlow)
        fake.model = FakeRoMaModel()
        result = fake.pair(torch.zeros(1, 3, 8, 16), torch.zeros(1, 3, 8, 16))
        self.assertEqual(result.forward.shape, (1, 2, 8, 16))
        self.assertEqual(result.backward.shape, (1, 2, 8, 16))
```

- [ ] **Step 2: Run the tests and verify the intended failures**

Run: `python -m unittest tests.test_romav2_flow -v`

Expected: FAIL because `prepare_romav2_source`, `prepare_romav2_checkpoint` and `RoMaFlowResult` do not exist.

- [ ] **Step 3: Implement explicit offline setup and named results**

```python
@dataclass(frozen=True)
class RoMaFlowResult:
    forward: torch.Tensor
    backward: torch.Tensor
    overlap_forward: torch.Tensor
    overlap_backward: torch.Tensor


def prepare_romav2_source(source_root: str | Path) -> Path:
    source = Path(source_root).expanduser().resolve() / "src"
    if not (source / "romav2" / "__init__.py").is_file():
        raise FileNotFoundError(f"RoMa v2 source must contain src/romav2: {source_root}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def prepare_romav2_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RoMa v2 checkpoint not found: {path}")
    if path.name != "romav2.0.1.pt" or path.parent.name != "checkpoints" or path.parent.parent.name != "hub":
        raise ValueError("RoMa v2 checkpoint must be .../hub/checkpoints/romav2.0.1.pt")
    torch.hub.set_dir(str(path.parent.parent))
    return path
```

Update `RoMaFlow.__init__` to call both helpers before importing `RoMaV2`, preserve `fast/base`, and return `RoMaFlowResult` from `pair()`.

- [ ] **Step 4: Run focused and existing compatibility tests**

Run: `python -m unittest tests.test_romav2_flow tests.test_flow_metrics_romav2 -v`

Expected: PASS; existing `warp_to_flow` and metrics behavior remains unchanged.

- [ ] **Step 5: Commit the adapter**

```bash
git add stereo_center/stereo_center/romav2_flow.py tests/test_romav2_flow.py
git commit -m "feat: make RoMa v2 loading offline and explicit"
```

### Task 2: Flow Diagnostics

**Files:**
- Create: `stereo_center/stereo_center/flow_diagnostics.py`
- Create: `tests/test_flow_diagnostics.py`

**Interfaces:**
- Consumes: RGB arrays `H,W,3`, forward/backward flow arrays `H,W,2`。
- Produces: `FlowDiagnostics(warped_source, photometric_error, cycle_error, valid)`。

- [ ] **Step 1: Write exact translation and out-of-bounds tests**

```python
def test_cycle_error_is_zero_for_inverse_translation():
    forward = np.zeros((6, 8, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 1
    backward[..., 0] = -1
    error, valid = forward_backward_error(forward, backward)
    np.testing.assert_allclose(error[valid], 0, atol=1e-6)
    self.assertFalse(valid[:, -1].any())


def test_backward_warp_does_not_replicate_border():
    source = np.arange(24, dtype=np.uint8).reshape(4, 6)
    flow = np.zeros((4, 6, 2), np.float32)
    flow[..., 0] = 10
    warped, valid = backward_warp(source, flow)
    self.assertFalse(valid.any())
    self.assertTrue((warped == 0).all())
```

- [ ] **Step 2: Verify the module is absent**

Run: `python -m unittest tests.test_flow_diagnostics -v`

Expected: FAIL with `ModuleNotFoundError: stereo_center.flow_diagnostics`.

- [ ] **Step 3: Implement coordinate-safe diagnostics**

```python
@dataclass(frozen=True)
class FlowDiagnostics:
    warped_source: np.ndarray
    photometric_error: np.ndarray
    cycle_error: np.ndarray
    valid: np.ndarray


def forward_backward_error(forward, backward):
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    map_x = xx + forward[..., 0]
    map_y = yy + forward[..., 1]
    sampled = cv2.remap(backward, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return np.linalg.norm(forward + sampled, axis=-1), valid
```

Implement `backward_warp()` and `compute_flow_diagnostics()` with finite-value checks and a shared in-bounds mask.

- [ ] **Step 4: Run diagnostics tests**

Run: `python -m unittest tests.test_flow_diagnostics -v`

Expected: PASS for identity, inverse translation, out-of-bounds and nonfinite rejection.

- [ ] **Step 5: Commit diagnostics**

```bash
git add stereo_center/stereo_center/flow_diagnostics.py tests/test_flow_diagnostics.py
git commit -m "feat: add optical flow consistency diagnostics"
```

### Task 3: Stable Video Visualization

**Files:**
- Create: `stereo_center/stereo_center/flow_visualization.py`
- Create: `tests/test_flow_visualization.py`

**Interfaces:**
- Consumes: BGR source/target frames, `H,W,2` flow, valid masks and one run-wide `max_magnitude_px`。
- Produces: five fixed-size BGR frames suitable for `cv2.VideoWriter`。

- [ ] **Step 1: Write visualization contract tests**

```python
def test_same_vector_has_same_color_across_frames():
    first = np.zeros((8, 8, 2), np.float32); first[..., 0] = 2
    second = first.copy(); second[0, 0, 0] = 20
    a = flow_to_color(first, max_magnitude_px=10)
    b = flow_to_color(second, max_magnitude_px=10)
    np.testing.assert_array_equal(a[4, 4], b[4, 4])


def test_nonfinite_flow_is_black():
    flow = np.zeros((4, 4, 2), np.float32)
    flow[1, 1] = np.nan
    image = flow_to_color(flow, max_magnitude_px=5)
    np.testing.assert_array_equal(image[1, 1], [0, 0, 0])


def test_report_frame_is_two_by_three_panels():
    panels = [np.zeros((20, 30, 3), np.uint8) for _ in range(6)]
    self.assertEqual(compose_report_frame(*panels).shape, (40, 90, 3))
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `python -m unittest tests.test_flow_visualization -v`

Expected: FAIL because the visualization module does not exist.

- [ ] **Step 3: Implement five frame renderers**

```python
@dataclass(frozen=True)
class FlowVisualizationFrames:
    color: np.ndarray
    magnitude: np.ndarray
    arrows: np.ndarray
    diagnostics: np.ndarray
    report: np.ndarray


def render_flow_frames(source_bgr, target_bgr, flow, diagnostics, max_magnitude_px):
    color = flow_to_color(flow, max_magnitude_px)
    magnitude = magnitude_to_heatmap(flow, max_magnitude_px)
    arrows = draw_flow_arrows(source_bgr, flow, step=32)
    diagnostic = diagnostic_frame(diagnostics)
    report = compose_report_frame(source_bgr, target_bgr, color, magnitude, arrows, diagnostic)
    return FlowVisualizationFrames(color, magnitude, arrows, diagnostic, report)
```

Use the RAFT/Middlebury color wheel, BGR output, fixed labels, fixed arrow grid and explicit `px/frame_stride` legend. Reject nonpositive `max_magnitude_px`; zero flow remains valid and renders white in the color view.

- [ ] **Step 4: Run visualization and diagnostics tests**

Run: `python -m unittest tests.test_flow_visualization tests.test_flow_diagnostics -v`

Expected: PASS and all frames retain stable dimensions.

- [ ] **Step 5: Commit visualization**

```bash
git add stereo_center/stereo_center/flow_visualization.py tests/test_flow_visualization.py
git commit -m "feat: add stable optical flow video visualization"
```

### Task 4: Evaluation Core and Exact Video Pair Counting

**Files:**
- Create: `stereo_center/stereo_center/flow_evaluation.py`
- Create: `tests/test_flow_evaluation.py`

**Interfaces:**
- Consumes: `FlowEvaluationConfig` and a model exposing `pair(previous, current) -> RoMaFlowResult`。
- Produces: memmaps, `FlowRunSummary`, sampled global P99, per-pair timing and proxy diagnostics。

- [ ] **Step 1: Write fake-model integration tests**

```python
def test_scan_uses_decodable_frames_not_metadata(self):
    path = write_test_video(frame_count=5, advertised_count_override=9)
    scan = scan_video(path)
    self.assertEqual(scan.decoded_frames, 5)
    self.assertEqual(pair_count(scan.decoded_frames, stride=1), 4)


def test_warmup_is_excluded_and_memmap_has_uv_layout(self):
    model = FakeBidirectionalFlow()
    summary = evaluate_video(model, self.video, self.outdir, frame_stride=1, warmup_pairs=1)
    self.assertEqual(model.calls, summary.pair_count + 1)
    forward = np.load(self.outdir / "flow_forward.npy", mmap_mode="r")
    self.assertEqual(forward.shape, (summary.pair_count, 16, 24, 2))
    self.assertEqual(len(summary.inference_seconds), summary.pair_count)
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `python -m unittest tests.test_flow_evaluation -v`

Expected: FAIL because `flow_evaluation.py` does not exist.

- [ ] **Step 3: Implement pre-scan, memmaps and deterministic P99 sampling**

```python
@dataclass(frozen=True)
class FlowEvaluationConfig:
    video: Path
    output_dir: Path
    frame_stride: int = 1
    warmup_pairs: int = 1
    sample_step: int = 16
    max_pairs: int | None = None


@dataclass(frozen=True)
class FlowRunSummary:
    decoded_frames: int
    pair_count: int
    width: int
    height: int
    fps: float
    flow_color_max_px: float
    inference_seconds: tuple[float, ...]
```

`scan_video()` must decode to EOF and reject fewer than `frame_stride + 1` frames. `evaluate_video()` allocates exact NPY memmaps, performs one unmeasured warm-up call, synchronizes CUDA around timed calls, writes both directions and overlaps, and samples `magnitude[::16, ::16]` from every pair for the run-wide P99.
When `max_pairs` is set, allocate and process exactly
`min(max_pairs, available_pair_count)` pairs; reject values below 1.

- [ ] **Step 4: Run core tests and the existing metric tests**

Run: `python -m unittest tests.test_flow_evaluation tests.test_flow_metrics_romav2 -v`

Expected: PASS; no CUDA or real RoMa weight is required by unit tests.

- [ ] **Step 5: Commit evaluation core**

```bash
git add stereo_center/stereo_center/flow_evaluation.py tests/test_flow_evaluation.py
git commit -m "feat: add standalone optical flow evaluation core"
```

### Task 5: CLI, Five MP4 Outputs and Atomic Completion Manifest

**Files:**
- Create: `stereo_center/scripts/run_flow_evaluation.py`
- Modify: `tests/test_flow_evaluation.py`

**Interfaces:**
- Consumes: CLI paths/settings and Task 1-4 APIs。
- Produces: `flow_color.mp4`, `flow_magnitude.mp4`, `flow_arrows.mp4`, `flow_diagnostics.mp4`, `flow_report.mp4`, NPY caches, `metrics.json`, `manifest.json`。

- [ ] **Step 1: Add CLI and writer failure tests**

```python
def test_cli_requires_video_and_existing_offline_assets(self):
    with self.assertRaises(SystemExit):
        parse_args([])


@patch("cv2.VideoWriter")
def test_failed_video_writer_does_not_mark_run_complete(self, writer_type):
    writer_type.return_value.isOpened.return_value = False
    with self.assertRaisesRegex(RuntimeError, "flow_color.mp4"):
        render_videos(self.config, self.summary)
    self.assertFalse((self.outdir / "manifest.json").exists())


def test_verify_outputs_rejects_short_video(self):
    write_test_video(self.outdir / "flow_color.mp4", frame_count=2)
    with self.assertRaisesRegex(RuntimeError, "frame count"):
        verify_video_outputs(self.outdir, expected_frames=4)
```

- [ ] **Step 2: Run the focused tests and observe failures**

Run: `python -m unittest tests.test_flow_evaluation -v`

Expected: FAIL because CLI parsing, rendering and output verification functions do not exist.

- [ ] **Step 3: Implement the CLI and two-pass rendering**

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["roma-v2"], default="roma-v2")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roma-source-root", type=Path, required=True)
    parser.add_argument("--roma-checkpoint", type=Path, required=True)
    parser.add_argument("--roma-setting", choices=["fast", "base"], default="fast")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--flow-color-max-px", type=float)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)
```

Write JSON to temporary sibling files and rename only after all five videos pass decode/frame-count checks. Set `manifest.json` field `status` to `complete` only after verification. Record source revision, checkpoint SHA-256, video path, decoded count, pair count, input fps, output dimensions, P99 scale, package versions and GPU name.

- [ ] **Step 4: Run the complete local CPU suite**

Run: `python -m unittest tests.test_romav2_flow tests.test_flow_diagnostics tests.test_flow_visualization tests.test_flow_evaluation tests.test_flow_metrics_romav2 -v`

Expected: PASS with no network access and no GPU dependency.

- [ ] **Step 5: Compile the entry point and commit**

Run: `python -m py_compile stereo_center/scripts/run_flow_evaluation.py stereo_center/stereo_center/flow_*.py stereo_center/stereo_center/romav2_flow.py`

Expected: exit code 0.

```bash
git add stereo_center/scripts/run_flow_evaluation.py stereo_center/stereo_center/flow_evaluation.py tests/test_flow_evaluation.py
git commit -m "feat: add RoMa v2 flow evaluation CLI"
```

### Task 6: A100 Smoke, Full 010031 Run and Local Video Delivery

**Files:**
- Create: `docs/2026-09-22-010031-RoMa-v2完整视频实验记录.md`

**Interfaces:**
- Consumes: committed local implementation, A100 `depth` conda environment, fixed 010031 source video and fixed RoMa v2 assets。
- Produces: verified full-video artifacts on A100 and downloaded copies under the local Desktop。

- [ ] **Step 1: Verify local branch and create a transfer bundle**

Run:

```bash
git status --short
git log -1 --oneline
git bundle create /tmp/CenterDepth-roma-flow-eval-20260922.bundle main
sha256sum /tmp/CenterDepth-roma-flow-eval-20260922.bundle
```

Expected: worktree clean; bundle contains the implementation commits.

- [ ] **Step 2: Upload through the bastion and fast-forward the isolated A100 checkout**

Run:

```bash
scp -O /tmp/CenterDepth-roma-flow-eval-20260922.bundle ziki:/tmp/CenterDepth-roma-flow-eval-20260922.bundle
ssh ziki 'git -C /home/opsuser/zata-dataset-tools-yhy/CenterDepth fetch /tmp/CenterDepth-roma-flow-eval-20260922.bundle main && git -C /home/opsuser/zata-dataset-tools-yhy/CenterDepth merge --ff-only FETCH_HEAD'
```

Expected: remote `main` equals local `main`; existing `zata-dataset-tools-yhy` files and runtime directories outside `CenterDepth` remain unchanged.

- [ ] **Step 3: Run A100 unit tests and a two-pair smoke test**

Run:

```bash
PYTHONPATH=/home/opsuser/zata-dataset-tools-yhy/CenterDepth/stereo_center:/home/opsuser/BothEyesDepth/third_party/RoMaV2/src \
/home/opsuser/miniconda3/envs/depth/bin/python -m unittest \
tests.test_romav2_flow tests.test_flow_diagnostics tests.test_flow_visualization \
tests.test_flow_evaluation tests.test_flow_metrics_romav2 -v
```

Then run the CLI with `--max-pairs 2` into
`/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-smoke-20260922-r1`.

Expected: tests pass, all five smoke MP4 files decode to exactly 2 frames, all flow values are finite, and manifest status is `complete`.

- [ ] **Step 4: Run the complete 010031 left video**

Run:

```bash
PYTHONPATH=/home/opsuser/zata-dataset-tools-yhy/CenterDepth/stereo_center:/home/opsuser/BothEyesDepth/third_party/RoMaV2/src \
/home/opsuser/miniconda3/envs/depth/bin/python \
/home/opsuser/zata-dataset-tools-yhy/CenterDepth/stereo_center/scripts/run_flow_evaluation.py \
  --backend roma-v2 \
  --video /home/opsuser/BothEyesDepth/dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_camera_left_part0001.mp4 \
  --output-dir /home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1 \
  --roma-source-root /home/opsuser/BothEyesDepth/third_party/RoMaV2 \
  --roma-checkpoint /home/opsuser/.cache/torch/hub/checkpoints/romav2.0.1.pt \
  --roma-setting fast \
  --frame-stride 1 \
  --warmup-pairs 1 \
  --device cuda
```

Expected: every decodable adjacent pair is processed; `manifest.json` reports `status=complete`; all five MP4 files have `pair_count` frames.

- [ ] **Step 5: Validate artifacts and record performance**

Check JSON for mean/median/P95/P99 latency, effective FPS, CUDA peak memory, motion magnitude P99, cycle-valid ratio and photometric MAE. Decode every MP4 to EOF and compare actual frames, fps and dimensions with manifest. Write the exact command, Git commit, model/weight revisions, timing table, artifact sizes and any anomalies to `docs/2026-09-22-010031-RoMa-v2完整视频实验记录.md`.

- [ ] **Step 6: Download the user-facing artifacts to Desktop**

Run:

```bash
mkdir -p /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/flow_color.mp4 /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/flow_magnitude.mp4 /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/flow_arrows.mp4 /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/flow_diagnostics.mp4 /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/flow_report.mp4 /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/metrics.json /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
scp -O ziki:/home/opsuser/BothEyesDepth/outputs/flow-eval/010031-romav2-fast-20260922-r1/manifest.json /Users/xupeihong/Desktop/010031-romav2-flow-eval-20260922-r1/
```

Expected: local SHA-256 values match A100 values; each local MP4 decodes fully and has the expected frame count.

- [ ] **Step 7: Commit the experiment record**

```bash
git add -f docs/2026-09-22-010031-RoMa-v2完整视频实验记录.md
git commit -m "docs: record 010031 RoMa v2 flow evaluation"
```
