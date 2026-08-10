# S²M² + SoftSplat 最小可运行工程

目标：验证“双目 → 中心视角 RGB + Depth”管线的可行性。输入为
VDEgo-C2 采集的双目视频帧（3840×1200 = 左右各 1920×1200 鱼眼），输出为
**中间虚拟相机视角的 RGB 图与深度图**。

## 管线

```
双目帧 ──► 鱼眼极线校正 ──► S²M² (disparity/occlusion/confidence)
                                    │
                                    ▼
                        Softmax Splatting 前向投影
                                    │
                                    ▼
                        Center RGB + Center Depth
```

## 目录结构

```
stereo_center/
├── stereo_center/
│   ├── calib.py          # VDEgo-C2 标定加载 + 鱼眼平行双目校正
│   ├── s2m2_inference.py # S²M² 模型加载与推理
│   ├── softsplat.py      # 纯 PyTorch Softmax Splatting + 中心视角合成
│   └── pipeline.py       # 组合管线
├── scripts/
│   ├── run_pipeline.py   # 单帧运行脚本
│   └── download_weights.sh
├── third_party/s2m2/     # S²M² 官方仓库（vendored）
├── weights/pretrain_weights/CH128NTR1.pth  # S 轻量权重
└── outputs/              # 运行结果
```

## 环境准备（当前目录）

```bash
# 在项目根目录（BothEyesDepth/）创建虚拟环境，复用已有的 torch 环境
/opt/miniconda3/envs/evograph-r1/bin/python -m venv --system-site-packages .venv

# 安装 s2m2（--no-deps：torch/opencv/numpy 已由基础环境提供）
.venv/bin/python -m pip install --no-build-isolation --no-deps -e stereo_center/third_party/s2m2

# 下载轻量权重（S 版）
./stereo_center/scripts/download_weights.sh
```

## 运行

```bash
cd stereo_center
../.venv/bin/python scripts/run_pipeline.py \
    --video ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/output.mp4 \
    --calib ../vdego-c2-48b749_2026-07-28_10-27-26_30fps/calibration.json \
    --frame 60 --scale 0.5 --outdir outputs/run_1
```

产物：

| 文件 | 说明 |
| --- | --- |
| `rect_left.png` / `rect_right.png` | 极线校正后的左右图 |
| `disparity.png` / `occlusion.png` / `confidence.png` | S²M² 输出 |
| `center_rgb.png` | 中心视角 RGB |
| `center_depth.npy` / `center_depth.png` | 中心视角深度（米） |
| `overview.png` | 2×3 总览图：左校正 | 中心RGB | 右校正 / 视差 | 中心深度 | 置信度 |

## 当前实现说明与限制

1. **校正**：采用“理想平行双目”假设（R=I, t=(B,0,0)），忽略相机间约 0.75°
   的小旋转；实测极线对齐误差中位数 ~0.9px，满足 S²M² 建议的 <2px。
   若需要更严格的对齐，可改用 S²M² 仓库自带的在线校正模块。
2. **SoftSplat**：官方 `sniklaus/softmax-splatting` 是 CUDA 算子，本工程在
   无 CUDA 的 Mac 上用纯 PyTorch scatter 实现了等价的前向软投影
   （`exp(weight)` 加权 + 归一化）。上 GPU 服务器后可直接替换为官方算子。
3. **右视差**：通过把左视差场前向投影到右图坐标得到（一步近似），
   在强遮挡边界可能有误差。
4. **CPU 推理**：S 模型在 960×600 上单帧约 1~3 分钟；上 GPU 可到实时量级。
5. **许可**：S²M² 与 Softmax Splatting 官方均为非商业研究/教育用途，
   进入产品前需处理许可。

## 验证结果（2026-08-10）

在 `vdego-c2-48b749_2026-07-28_10-27-26_30fps/output.mp4` 上运行
（S 模型、scale=0.5，即 960×600，CPU 单帧 ~9~10s）：

| 帧 | S²M² 平均置信度 | 中心视角有效像素 | 深度范围 | 备注 |
| --- | --- | --- | --- | --- |
| 60 | 0.960 | 98.2% | 0.06~29.94 m | 输出见 `outputs/run_1/` |
| 150 | 0.956 | 98.4% | 0.06~29.94 m | 输出见 `outputs/run_2/` |
| 250 | 0.853 | 98.0% | 0.07~29.94 m | 输出见 `outputs/run_3/` |

交叉验证：
- S²M² 视差与 OpenCV StereoBM 中位数比值 1.01、相关系数 0.65，两者一致；
- 校正后极线对齐误差中位数 ~0.9px（SIFT+RANSAC 评估）；
- 左右视图前向投影到中心后相互吻合（右流向符号 +d/2 优于 -d/2）。

注意：该视频为手持近距采集（相邻帧差异大、深度中位数约 0.2~0.5 m），
深度绝对尺度与采集场景有关；建议后续用已知距离场景标定验证深度精度。
