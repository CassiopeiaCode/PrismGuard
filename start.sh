#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".env" ]; then
    echo "错误: 未找到 .env 文件"
    exit 1
fi

BIN="${1:-./target/release/Prismguand-Rust}"

if [ ! -x "$BIN" ]; then
    echo "错误: 可执行文件不存在或不可执行: $BIN"
    echo "请先以低优先级构建，例如: nice -n 19 cargo build --release -j 1"
    exit 1
fi

echo "以 nice=19 启动 Prismguand-Rust: $BIN"
exec nice -n 19 "$BIN"
