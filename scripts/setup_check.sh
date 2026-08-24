#!/usr/bin/env bash
# setup_check.sh — preflight checks for a DGX Spark before benchmarking.
set -uo pipefail
ok(){ printf "  \033[32mok\033[0m    %s\n" "$1"; }
warn(){ printf "  \033[33mwarn\033[0m  %s\n" "$1"; }
bad(){ printf "  \033[31mFAIL\033[0m  %s\n" "$1"; }

echo
echo "== kernel-ladder preflight =="
echo

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then ok "aarch64 (Grace CPU) — as expected on Spark"
else warn "this box is $ARCH, not aarch64. Not a DGX Spark; tile sizes and the 99KB shared-memory assumptions here are tuned for GB10."; fi

if command -v nvidia-smi >/dev/null 2>&1; then
  NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU: ${NAME:-unknown}  driver ${DRV:-unknown}"

  # GB10 can pin at 721 MHz with no throttle reason; only a power cycle clears it.
  SM=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "${SM:-}" ] && [ "$SM" != "[N/A]" ]; then
    if [ "$SM" -lt 900 ] 2>/dev/null; then
      bad "SM clock is ${SM} MHz — the known GB10 low-clock state. POWER CYCLE before benchmarking."
    else ok "SM clock ${SM} MHz"; fi
  fi

  if nvidia-smi -q -d SUPPORTED_CLOCKS 2>/dev/null | grep -qi 'N/A\|not supported'; then
    warn "supported clocks report N/A — you CANNOT lock clocks on GB10."
    warn "  The harness records SM clock before and after every measurement instead."
  fi
else
  bad "nvidia-smi not found — no GPU visible"
fi

if command -v nvcc >/dev/null 2>&1; then
  V=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  MAJ=${V%%.*}; MIN=$(echo "$V" | cut -d. -f2)
  if [ "$MAJ" -gt 12 ] || { [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 9 ]; }; then
    ok "CUDA $V (sm_121 needs >= 12.9)"
  else
    bad "CUDA $V is too old for sm_121. Need >= 12.9; DGX OS 7.5 ships 13.0.2."
  fi
else
  bad "nvcc not found — check PATH (usually /usr/local/cuda/bin)"
fi

if command -v ncu >/dev/null 2>&1; then
  ok "Nsight Compute (ncu) present"
  if [ "$(id -u)" -ne 0 ] && ! grep -rqs 'NVreg_RestrictProfilingToAdminUsers=0' /etc/modprobe.d/ 2>/dev/null; then
    warn "ncu may fail with ERR_NVGPUCTRPERM as a non-root user. Fix permanently:"
    warn "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiler.conf"
    warn "  sudo update-initramfs -u && sudo reboot"
  fi
else
  warn "ncu not found — profiling rungs will not work. It ships with the CUDA toolkit."
fi
command -v nsys >/dev/null 2>&1 && ok "Nsight Systems (nsys) present" || warn "nsys not found"
command -v compute-sanitizer >/dev/null 2>&1 && ok "compute-sanitizer present" \
  || warn "compute-sanitizer not found — needed for make sanitize"

warn "nsys unified-memory tracing (--cuda-um-*) is unsupported on GB10. Known NVIDIA limitation."

# GPU and page cache share one 128GB pool; a full cache can cause spurious CUDA OOM.
if [ -r /proc/meminfo ]; then
  CACHED=$(awk '/^Cached:/{print int($2/1024/1024)}' /proc/meminfo)
  AVAIL=$(awk '/^MemAvailable:/{print int($2/1024/1024)}' /proc/meminfo)
  ok "host memory: ${AVAIL} GB available, ${CACHED} GB in page cache"
  if [ "${CACHED:-0}" -gt 20 ]; then
    warn "page cache holds ${CACHED} GB of the shared pool. Before benchmarking:"
    warn "  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${N:-0}" -gt 0 ]; then
    warn "$N other process(es) are using the GPU. They will contend for the same"
    warn "  273 GB/s of memory bandwidth and fatten the latency tail."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/        /'
  else
    ok "no other compute processes on the GPU"
  fi
fi

echo
echo "next: make probe   (which tensor-core instructions sm_121 actually supports)"
echo "      make build && make run-small"
echo
