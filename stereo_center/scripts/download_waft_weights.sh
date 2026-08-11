#!/usr/bin/env bash
# 下载 WAFT-Stereo 零样本权重（HuggingFace MemorySlices/WAFT-Stereo）。
# 用法: ./download_waft_weights.sh [DAv2L-5|DAv2B-4|DAv2S-4]   (默认 DAv2L-5)
set -euo pipefail
cd "$(dirname "$0")/../.."  # 仓库根目录
mkdir -p weights/waft

MODEL="${1:-DAv2L-5}"
case "$MODEL" in
  DAv2S-4|DAv2B-4|DAv2L-5) REL="SynLarge/${MODEL}.pth" ;;
  *) echo "未知模型: $MODEL（可选 DAv2S-4 / DAv2B-4 / DAv2L-5）" >&2; exit 1 ;;
esac
OUT="weights/waft/${MODEL}.pth"

URLS=(
  "https://huggingface.co/MemorySlices/WAFT-Stereo/resolve/main/${REL}"
  "https://hf-mirror.com/MemorySlices/WAFT-Stereo/resolve/main/${REL}"
)
for u in "${URLS[@]}"; do
  echo "尝试下载: $u"
  if curl -fL --retry 2 -o "$OUT" "$u"; then
    echo "OK: $OUT"
    exit 0
  fi
done

echo "直连与镜像均失败，尝试本地代理 http://127.0.0.1:7897 ..."
if curl -x http://127.0.0.1:7897 -fL --retry 2 -o "$OUT" "${URLS[0]}"; then
  echo "OK: $OUT (via proxy)"
  exit 0
fi

echo "全部下载源失败: $OUT" >&2
exit 1
