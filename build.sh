#!/usr/bin/env bash
# build.sh — fetch deps, then compile the runner and the leverage tool.
#
#   ./build.sh              # ensure deps + build both binaries
#   ./build.sh --local      # snapshot ../TTTN working tree as the dep, then build
set -e
cd "$(dirname "$0")"

if [[ "${1:-}" == "--local" ]]; then
    ./fetch_deps.sh --local "${2:-../TTTN}"
else
    ./fetch_deps.sh
fi

FLAGS="-std=c++20 -O3 -mcpu=apple-m3 -Wall -Wextra -Wpedantic \
       -Wno-deprecated-declarations -fconstexpr-steps=67108864 -I."

echo "==> building nanda_grokking..."
g++ $FLAGS nanda_grokking.cpp -framework Accelerate -o nanda_grokking

echo "==> building nanda_leverage..."
g++ $FLAGS nanda_leverage.cpp -framework Accelerate -o nanda_leverage

echo "==> building nanda_jlens..."
g++ $FLAGS nanda_jlens.cpp -framework Accelerate -o nanda_jlens

echo "==> building nanda_jspace..."
g++ $FLAGS nanda_jspace.cpp -framework Accelerate -o nanda_jspace

echo "==> done: ./nanda_grokking  ./nanda_leverage  ./nanda_jlens  ./nanda_jspace"
