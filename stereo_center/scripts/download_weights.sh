#!/usr/bin/env bash
# 下载 S²M² 轻量权重（S 版 CH128NTR1.pth）
set -euo pipefail
cd "$(dirname "$0")/../.."  # 仓库根目录
mkdir -p weights/pretrain_weights

URL="https://huggingface.co/minimok/s2m2/resolve/main/CH128NTR1.pth"
OUT="weights/pretrain_weights/CH128NTR1.pth"

if curl -fL --retry 2 -o "$OUT" "$URL"; then
    echo "OK: $OUT"
else
    echo "直连失败，尝试本地代理 http://127.0.0.1:7897 ..."
    curl -x http://127.0.0.1:7897 -fL --retry 2 -o "$OUT" "$URL"
    echo "OK: $OUT (via proxy)"
fi
