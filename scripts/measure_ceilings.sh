#!/usr/bin/env bash
# Re-measure the DRAM bandwidth and fp32 peak ceilings assumed by gb10.hpp, report.py, and sweep.py.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=${BUILD:-build}
ARCH=${ARCH:-sm_121}
mkdir -p "$BUILD"

command -v nvcc >/dev/null || { echo "nvcc not found"; exit 1; }

nvcc -O3 -arch="$ARCH" -o "$BUILD/measure_ceilings" scripts/measure_ceilings.cu
exec "$BUILD/measure_ceilings"
