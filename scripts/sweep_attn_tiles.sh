#!/usr/bin/env bash
# Sweep (Br, Bc) tile sizes for 02_flash_tiled: gate, build, verify, time.
# Usage: KBR_LIST="32 64 128" KBC_LIST="32 64 128" ./scripts/sweep_attn_tiles.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ARCH=${ARCH:-sm_121}
NVCC=${NVCC:-nvcc}
KBR_LIST=${KBR_LIST:-"32 64 128"}
KBC_LIST=${KBC_LIST:-"32 64 128"}
VERIFY_SIZE=${VERIFY_SIZE:-4x8x256x64}
TIME_SIZE=${TIME_SIZE:-4x8x2048x64}
ITERS=${ITERS:-30}
OUT=${OUT:-bench/results/sweep_attn.jsonl}
BUILD_DIR=${BUILD_DIR:-build/sweep_attn}
DRY_RUN=${DRY_RUN:-0}
THREADS=128     # fixed in 02_flash_tiled.cu
MAXD=64
SMEM_CAP=$((99 * 1024))

if grep -qE 'flash_tiled\s*=\s*false' src/attention/kernels.cuh; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "02_flash_tiled is disabled in src/attention/kernels.cuh."
        echo "Enable it before sweeping, or FORCE=1 to sweep anyway."
        exit 2
    fi
fi

command -v "$NVCC" >/dev/null || { echo "nvcc not found"; exit 1; }
mkdir -p "$BUILD_DIR" "$(dirname "$OUT")"

n_ok=0 n_gate=0 n_fail=0

emit_failure() {  # status message kbr kbc smem
    [ "$DRY_RUN" = "1" ] && return
    printf '{"phase":"attention_sweep","kernel":"02_flash_tiled","status":"%s","message":"%s","kbr":%d,"kbc":%d,"threads":%d,"smem_bytes":%d}\n' \
        "$1" "$2" "$3" "$4" "$THREADS" "$5" >> "$OUT"
}

for KBR in $KBR_LIST; do
for KBC in $KBC_LIST; do
    CFG="kbr${KBR}_kbc${KBC}"
    SMEM=$(( (KBR * MAXD + 2 * KBC * MAXD + KBR * KBC) * 4 ))

    if [ $((THREADS % KBR)) -ne 0 ] 2>/dev/null || [ "$KBR" -gt "$THREADS" ]; then
        printf '%-18s GATE: threads %% Br != 0\n' "$CFG"
        emit_failure "gate_threads" "128 threads cannot split $KBR rows" "$KBR" "$KBC" "$SMEM"
        n_gate=$((n_gate + 1)); continue
    fi
    TPR=$((THREADS / KBR))
    if [ $((MAXD % TPR)) -ne 0 ] || [ $((KBC % TPR)) -ne 0 ]; then
        printf '%-18s GATE: D or Bc not divisible by threads-per-row %d\n' "$CFG" "$TPR"
        emit_failure "gate_divisibility" "D/Bc not divisible by tpr=$TPR" "$KBR" "$KBC" "$SMEM"
        n_gate=$((n_gate + 1)); continue
    fi
    if [ "$SMEM" -gt "$SMEM_CAP" ]; then
        printf '%-18s GATE: %d KB smem over the 99 KB cap\n' "$CFG" $((SMEM / 1024))
        emit_failure "gate_smem" "$((SMEM / 1024)) KB over cap" "$KBR" "$KBC" "$SMEM"
        n_gate=$((n_gate + 1)); continue
    fi

    printf '%-18s %3d KB smem ... ' "$CFG" $((SMEM / 1024))
    if [ "$DRY_RUN" = "1" ]; then echo "would build"; continue; fi

    BIN="$BUILD_DIR/attention_$CFG"
    LOG="$BUILD_DIR/$CFG.build.log"
    if ! "$NVCC" -O3 -std=c++17 -arch="$ARCH" --threads 0 \
            -DKBR="$KBR" -DKBC="$KBC" -Xptxas -v \
            src/attention/*.cu -lcublas -o "$BIN" > "$LOG" 2>&1; then
        msg="$(grep -m1 -E 'static assertion|static_assert|error:' "$LOG" | head -c 200)"
        printf 'COMPILE ERROR\n    %s\n' "${msg:-see $LOG}"
        emit_failure "compile_error" "${msg:-nvcc failed}" "$KBR" "$KBC" "$SMEM"
        n_fail=$((n_fail + 1)); continue
    fi
    SPILL="$(grep -oE '[0-9]+ bytes spill stores' "$LOG" | sort -rn | head -1 | grep -oE '^[0-9]+')"
    SPILL="${SPILL:-0}"
    REGS="$(grep -A2 'flash_tiled_kernel' "$LOG" | grep -oE 'Used [0-9]+ registers' | grep -oE '[0-9]+' | head -1)"
    REGS="${REGS:-0}"

    if ! "$BIN" --size "$VERIFY_SIZE" --only 02_flash_tiled --iters 2 \
            --out /dev/null 2>/dev/null | grep -q ' ok'; then
        printf 'WRONG at %s\n' "$VERIFY_SIZE"
        emit_failure "verify_failed" "oracle mismatch at $VERIFY_SIZE" "$KBR" "$KBC" "$SMEM"
        n_fail=$((n_fail + 1)); continue
    fi
    if ! "$BIN" --size "$VERIFY_SIZE" --causal --only 02_flash_tiled --iters 2 \
            --out /dev/null 2>/dev/null | grep -q ' ok'; then
        printf 'WRONG (causal) at %s\n' "$VERIFY_SIZE"
        emit_failure "verify_failed_causal" "oracle mismatch causal at $VERIFY_SIZE" "$KBR" "$KBC" "$SMEM"
        n_fail=$((n_fail + 1)); continue
    fi

    # --no-verify is safe: this exact binary was just verified.
    TMP="$BUILD_DIR/$CFG.jsonl"
    : > "$TMP"
    "$BIN" --size "$TIME_SIZE" --only 02_flash_tiled --no-verify \
        --iters "$ITERS" --out "$TMP" >/dev/null 2>&1

    python3 - "$TMP" "$OUT" "$KBR" "$KBC" "$REGS" "$SPILL" "$SMEM" <<'PY'
import json, sys
tmp, out, kbr, kbc, regs, spill, smem = sys.argv[1:]
with open(tmp) as fh:
    rec = json.loads(fh.readlines()[-1])
rec.update(sweep="attn_tiles", kbr=int(kbr), kbc=int(kbc),
           registers_per_thread=int(regs), spill_bytes=int(spill),
           smem_bytes=int(smem))
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(f"{rec['p50_ms']:8.3f} ms  {rec['gflops_p50']:9.1f} GFLOP/s  "
      f"{regs} regs, {spill} spill B")
PY
    n_ok=$((n_ok + 1))
done
done

echo
echo "sweep done: $n_ok timed, $n_gate gated, $n_fail failed -> $OUT"
