#!/usr/bin/env bash
# Profile one kernel with Nsight Compute.
# Usage: ./scripts/profile.sh <kernel-name> [size]
# Default size 2048: ncu replays kernels many times, so a full-set profile at 4096 takes minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${1:?usage: profile.sh <kernel-name> [size]}
SIZE=${2:-2048}
BUILD=${BUILD:-build}
OUT="bench/results/ncu_${K}_${SIZE}"

command -v ncu >/dev/null || { echo "ncu not found (ships with the CUDA toolkit)"; exit 1; }

# Perf counters are admin-only under RmProfilingAdminOnly=1; use the passwordless ncu sudo rule if installed.
NCU=(ncu)
if grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null; then
  if sudo -n /usr/local/cuda/bin/ncu --version >/dev/null 2>&1; then
    NCU=(sudo -n /usr/local/cuda/bin/ncu)
  else
    echo "warning: profiling is admin-restricted and no passwordless ncu sudo rule found"
  fi
fi

echo "profiling $K at ${SIZE}^3 ..."
echo

# --iters 1: ncu does its own replaying, so extra harness iterations would profile identical launches.
"${NCU[@]}" --set full \
    --replay-mode kernel \
    --export "$OUT" --force-overwrite \
    --kernel-name-base demangled \
    "$BUILD/ladder" --size "$SIZE" --only "$K" --warmup 1 --iters 1 --no-verify --out /dev/null

echo
echo "=============================================================="
echo " METRIC REFERENCE"
echo "=============================================================="
ncu --import "$OUT.ncu-rep" --page details --section SpeedOfLight 2>/dev/null | head -40 || true

cat <<'GUIDE'

Key metrics by optimization concern:

coalescing
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
    ~32 = fully uncoalesced strided fp32; ~4 = perfectly coalesced;
    ~1 = warp-uniform broadcast. The counter averages over all global loads.
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   total sectors fetched

memory traffic (GB10 has NO dram__* counters; use the L2 substitutes)
  lts__t_sectors_op_read.sum                L1->L2 read traffic (32B sectors)
  lts__t_sectors_op_read_lookup_miss.sum    ~DRAM reads, modulo the 24 MB L2
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum   smem conflicts

register tiling / occupancy
  sm__throughput.avg.pct_of_peak_sustained_elapsed          overall SoL
  launch__registers_per_thread                              occupancy driver
  sm__warps_active.avg.pct_of_peak_sustained_active         occupancy
  smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active   FMA pipe

vectorization
  make sass counts load/store widths directly: LDG.E.128 / STG.E.128 vs .32

tensor cores (GB10 has no hmma-cycles metric)
  sm__inst_executed_pipe_tensor.sum
    mma instruction count; matches MNK/(16*8*16) exactly when every MAC runs
    on the tensor pipe. Near zero means the kernel fell back to FMAs.

SpeedOfLight triage: Memory% high / Compute% low = data-movement bound;
Compute% high = arithmetic bound; both low = latency bound (raise occupancy
or work per thread).

GB10 caveat: nsys unified-memory tracing (--cuda-um-*) is unsupported on
this part; ncu itself works.
GUIDE
echo
echo "full report: ncu-ui $OUT.ncu-rep"
