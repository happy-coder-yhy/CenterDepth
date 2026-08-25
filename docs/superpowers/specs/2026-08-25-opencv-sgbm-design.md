# OpenCV StereoSGBM 设计

## 目标

将 OpenCV StereoSGBM 接入现有深度视频管线，作为无需权重和 GPU 的经典左参考立体匹配基线。

## 范围

- 新增独立后端名 `opencv_sgbm`，不改变 `opencv_bm`、WAFT、LAS2、FFS 或 SAV 的行为。
- 仅支持 `--output-view left --bi 0`，因此不进入光度对齐、中心视角融合或遮挡补洞路径。
- 复用现有视频解码、标定校正、深度换算、深度着色、视频写盘和计时结构。
- 生成 `opencv_sgbm_timing.json`，记录实际参数、逐批记录、前向总耗时、平均每帧耗时和 FPS。

## 架构

`opencv_sgbm_inference.py` 只负责参数校验、创建 `cv2.StereoSGBM` 和逐帧 CPU 匹配。`stereo_backend.py` 通过既有的 `load` 和 `run` 分发该模块。`run_depth_video.py` 负责 CLI、左单向模式约束和将 SGBM 参数写入已有 timing/stats 产物。

默认使用 OpenCV 的 `StereoSGBM_MODE_SGBM_3WAY`。它比 BM 使用多路径代价聚合，通常能得到更连续的视差；代价是 CPU 前向更慢。默认平滑项按 OpenCV 建议由块大小和 RGB 通道数计算：`P1=8*3*block_size^2`，`P2=32*3*block_size^2`，并允许用户显式覆盖。

## 验证

- 合成水平平移纹理图验证 SGBM 恢复已知视差。
- 验证非法参数、后端注册、左单向约束、参数序列化和 timing 文件名。
- 本地对 12-30 视频处理少量帧，验证视频和 JSON 产物可读且不运行中心视角阶段。

