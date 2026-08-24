#!/usr/bin/env bash
# run_ladder.sh — run the ladder under repeatable conditions; use instead of ./build/ladder directly.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=${BUILD:-build}
[ -x "$BUILD/ladder" ] || { echo "no $BUILD/ladder — run 'make build' first"; exit 1; }

# Drop the page cache: on Spark the GPU and page cache share one 128GB pool; non-fatal without sudo.
if [ "$(id -u)" -eq 0 ]; then
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "page cache dropped"
elif sudo -n true 2>/dev/null; then
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && echo "page cache dropped"
else
  echo "note: could not drop page cache (no passwordless sudo). Results will be noisier."
fi

# Record driver version, clocks, and git state alongside the results.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p bench/results
{
  echo "timestamp: $STAMP"
  echo "host: $(hostname)  arch: $(uname -m)  kernel: $(uname -r)"
  command -v nvcc >/dev/null && echo "cuda: $(nvcc --version | grep -oP 'release \K[0-9.]+')"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,clocks.sm,power.draw,temperature.gpu --format=csv
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
  git diff --quiet 2>/dev/null || echo "git: WORKING TREE DIRTY — results may not match any commit"
} > "bench/results/env_${STAMP}.txt"
echo "environment recorded -> bench/results/env_${STAMP}.txt"

# Warm the GPU so cold clocks don't penalize whichever rung runs first.
"$BUILD/ladder" --size 512 --iters 5 --no-verify --out /dev/null >/dev/null 2>&1 || true

exec "$BUILD/ladder" "$@"
