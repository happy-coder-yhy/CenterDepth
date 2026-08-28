# 双目鱼眼立体匹配模型对比

面向研究组内部的复现实验工程：在统一的双目鱼眼校正、视频处理和统计口径下，对比不同立体匹配后端的深度质量、速度、显存占用与时间稳定性。

项目以**左相机视角深度视频**作为模型对比的标准输出；对于需要虚拟中间相机视角的任务，可额外启用双向视差和中心视角合成，输出中心 RGB 与米制深度。

## 统一管线

```
输入双目视频 / 独立左右视频 + 标定
                │
                ▼
     PTS 同步（独立双视频时） + 鱼眼极线校正
                │
                ▼
     立体匹配后端（可批处理的视差推理）
                │
        ┌───────┴────────┐
        ▼                ▼
左视角深度视频      双向视差 + 中心视角合成（可选）
                │
                ▼
  深度着色、视频编码、时间与峰值显存统计
```

主入口为 `stereo_center/scripts/run_depth_video.py`。它统一完成视频解码、鱼眼校正、模型推理、深度可视化、视频写盘及统计记录；不同模型通过 `--stereo-backend` 切换。

## 支持后端

| 后端参数 | 模型/方法 | 批处理 | 中心视角合成 | 说明 |
| --- | --- | ---: | ---: | --- |
| `waft` | WAFT | 是 | 是 | 当前重点对比的深度立体匹配模型，支持可选时序初始化。 |
| `s2m2` | S2M2 | 否 | 是 | 早期中心视角合成基线。 |
| `las2` | LiteAnyStereo 2 | 是 | 是 | 深度立体匹配后端。 |
| `ffs` | Fast Foundation Stereo | 是 | 是 | 支持 `pytorch1` 或 `triton` 代价体后端。 |
| `stereonet` | StereoNet | 否 | 否 | 仅支持左视角、单向视差。 |
| `opencv_bm` | OpenCV StereoBM | 否 | 否 | 经典块匹配基线，仅支持左视角、单向视差。 |
| `opencv_sgbm` | OpenCV StereoSGBM | 否 | 否 | 经典半全局匹配基线，仅支持左视角、单向视差。 |

除明确仅左视角的后端外，`--bi 0 --output-view left` 是模型横向对比的推荐配置。中心视角实验需使用支持双向视差的后端，并设置 `--bi 1 --output-view center`。

## 数据与标定

支持两类输入：

| 数据类型 | 必需参数 | 标定 |
| --- | --- | --- |
| VDEgo-C2 SBS 双鱼眼视频 | `--video output.mp4` | `--calib calibration.json` |
| Orbbec 独立左右视频 | `--video left.mp4 --video-right right.mp4 --left-pts left_pts.csv --right-pts right_pts.csv` | `--calib calibration_camera.yaml` |

- VDEgo-C2 视频为左右并排的双鱼眼视频；管线会分离左右画面并进行鱼眼极线校正。
- Orbbec 的左右视频应使用采集时对应的 PTS CSV 配对。管线按最近时间戳匹配，避免仅依赖容器帧号造成左右不同步。
- 校正输出大小由 `--scale` 控制。`scale=0.5` 是当前标准实验尺度；更高尺度会显著增加模型前向时间和显存。

## 环境与权重

在项目根目录运行。Python 环境需包含 PyTorch、CUDA（GPU 实验）、OpenCV、NumPy、PyYAML 等依赖，并安装目标后端所需的第三方代码与依赖。

```bash
cd /path/to/BothEyesDepth
./.venv/bin/python stereo_center/scripts/run_depth_video.py --help
```

权重不纳入 Git。请将权重放在仓库根目录 `weights/` 下，或用 `--weights` 显式传入权重目录/文件。各后端的默认查找位置如下：

| 后端 | 默认权重位置 |
| --- | --- |
| WAFT | `weights/waft/` |
| S2M2 | `weights/pretrain_weights/` |
| LAS2 | `weights/las2/` 或 `weights/pretrain_weights/` |
| FFS | `weights/fast_foundation_stereo/` 或 `weights/pretrain_weights/` |
| StereoNet | `weights/stereonet/` |

FFS 还需要 Fast-FoundationStereo 源码。若不在默认位置，可通过 `--ffs-root` 指定；LAS2 和 StereoNet 分别可通过 `--las-root`、`--stereonet-root` 指定源码目录。

## 标准复现实验

以下命令均从仓库根目录执行，输出目录应具有可识别的模型、尺度、批大小和数据集名称。

### WAFT 左视角基准

适用于 VDEgo-C2 SBS 视频的标准模型对比。该模式不进行中心视角融合。

```bash
./.venv/bin/python stereo_center/scripts/run_depth_video.py \
  --stereo-backend waft \
  --video dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/output.mp4 \
  --calib dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/calibration.json \
  --model-type DAv2S-4 \
  --weights weights/waft \
  --scale 0.5 --batch-size 8 --iters 5 \
  --bi 0 --output-view left --left-vis-mode paper \
  --save-frames-every 0 \
  --outdir outputs/waft_s_s05_b8_left
```

### FFS 左视角基准

FFS 可选 `triton` 代价体后端。以下命令使用此前实验中验证过的批处理配置；请根据显存余量调整 `--batch-size`。

```bash
./.venv/bin/python stereo_center/scripts/run_depth_video.py \
  --stereo-backend ffs \
  --video dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/output.mp4 \
  --calib dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/calibration.json \
  --weights weights/fast_foundation_stereo/23-36-37 \
  --ffs-volume-backend triton \
  --scale 0.5 --batch-size 12 --iters 5 \
  --bi 0 --output-view left --left-vis-mode paper \
  --save-frames-every 0 \
  --outdir outputs/ffs_triton_s05_b12_left
```

### Orbbec 独立双视频实验

使用 PTS CSV 和 YAML 标定。下面以左视角 FFS 为例，替换为其他后端时保留输入、标定、尺度和输出视角参数即可。

```bash
./.venv/bin/python stereo_center/scripts/run_depth_video.py \
  --stereo-backend ffs \
  --video dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_camera_left_part0001.mp4 \
  --video-right dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_camera_right_part0001.mp4 \
  --left-pts dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_camera_left_part0001_pts.csv \
  --right-pts dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_camera_right_part0001_pts.csv \
  --calib dataset/abzg/Orbbec_Ego_AZER764001D_19700101_010031/Orbbec_Ego_AZER764001D_19700101_010031_calibration_camera.yaml \
  --weights weights/fast_foundation_stereo/23-36-37 \
  --ffs-volume-backend triton \
  --scale 0.5 --batch-size 12 --iters 5 \
  --bi 0 --output-view left --left-vis-mode paper \
  --save-frames-every 0 \
  --outdir outputs/ffs_triton_s05_b12_orbbec_010031
```

### 中心视角合成

中心视角需要左右双向视差，以处理虚拟相机视角下的遮挡与补洞。以下以 WAFT 为例：

```bash
./.venv/bin/python stereo_center/scripts/run_depth_video.py \
  --stereo-backend waft \
  --video dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/output.mp4 \
  --calib dataset/vdego-c2-48b749_2026-07-28_12-30-24_30fps/calibration.json \
  --model-type DAv2L-5 --weights weights/waft \
  --scale 0.5 --batch-size 8 --iters 5 \
  --bi 1 --output-view center \
  --save-frames-every 0 \
  --outdir outputs/waft_l_s05_b8_center
```

## 输出与统计

每个 `--outdir` 至少包含：

| 文件 | 说明 |
| --- | --- |
| `depth_video.mp4` | 深度着色视频；默认使用全片固定的对数米制色阶。 |
| `colorbar.png` | 深度视频对应的颜色标尺。 |
| `stats.json` | 端到端实验配置、逐阶段耗时、平均每帧耗时和峰值显存。 |
| `<backend>_timing.json` | 后端专属的逐批次推理时间，例如 `waft_timing.json`、`ffs_timing.json`。 |
| `frame_*.png` | 仅在 `--save-frames-every` 大于 0 时输出的采样帧。 |

`stats.json` 的关键字段包括：

- `total_seconds`：脚本主处理耗时。
- `end_to_end_seconds`：含模型加载的端到端耗时。
- `avg_seconds_per_frame`：端到端耗时除以处理帧数。
- `peak_gpu_memory_gib`：PyTorch 记录的峰值保留显存，单位 GiB。
- `stage_stereo_forward_seconds`：模型前向耗时。
- `stage_stereo_pipeline_seconds`：模型输入准备、前向和输出处理总耗时。

## 复现实验建议

1. 比较模型时固定数据、帧范围、`scale`、`batch-size`、输出视角和可视化模式。
2. 左视角基准固定使用 `--bi 0 --output-view left`，避免中心合成与双向推理掩盖模型本身的成本。
3. 中心视角实验单独记录，明确标注 `--bi 1`、融合和补洞阶段耗时。
4. 同时保留 `stats.json` 与后端 timing JSON；不要只用视频观感判断性能。
5. 时间平滑可能降低闪烁，也可能引入拖影，应作为独立消融实验，而非默认配置。

## 限制与许可

- 立体匹配依赖标定和左右同步质量；鱼眼校正、PTS 配对误差或低纹理区域都会影响深度结果。
- 经典 OpenCV 后端仅作为基线，其遮挡区、低纹理区和鱼眼边缘质量通常弱于深度模型。
- 中心视角合成会引入额外的双向推理、光度对齐、前向投影与补洞成本，不应与左视角基准直接比较总耗时。
- 各模型和第三方依赖保留其原始许可证。实验、发布或产品化前应分别核对 WAFT、S2M2、LiteAnyStereo、Fast-FoundationStereo、StereoNet、OpenCV 与 Softmax Splatting 的许可条款。
