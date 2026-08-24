#!/usr/bin/env bash
# sweep_tiles.sh — sweep rung 4 tile shapes, timing each and writing one JSONL record per config for bench/sweep.py.
# Usage: [BM_LIST="64 128"] [SIZE=2048 ITERS=30 OUT=...] [DRY_RUN=1] scripts/sweep_tiles.sh
# Then:  python3 bench/sweep.py bench/results/sweep_tiles.jsonl

set -uo pipefail  # no `set -e`: the sweep must keep going past failing configs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Powers of two only: other tile sizes break the divisibility gates and rung-5 vector-load alignment.
BM_LIST="${BM_LIST:-64 128}"
BN_LIST="${BN_LIST:-64 128}"
BK_LIST="${BK_LIST:-8 16}"
TM_LIST="${TM_LIST:-4 8}"
TN_LIST="${TN_LIST:-4 8}"

SIZE="${SIZE:-4096}"          # timed problem size
ITERS="${ITERS:-50}"          # measured iterations per config
WARMUP="${WARMUP:-15}"

# Each config is verified once at this small size, then timed at SIZE with verification off.
VERIFY_SIZE="${VERIFY_SIZE:-512}"

OUT="${OUT:-bench/results/sweep_tiles.jsonl}"
BUILD_DIR="${BUILD_DIR:-build/sweep}"
ARCH="${ARCH:-sm_121}"        # GB10
NVCC="${NVCC:-nvcc}"
DRY_RUN="${DRY_RUN:-0}"

# GB10 hard limits. Keep in sync with src/common/gb10.hpp.
MAX_SMEM_BYTES=$((99 * 1024))
MAX_THREADS=1024

mkdir -p "$BUILD_DIR" "$(dirname "$OUT")"

say() { printf '%s\n' "$*"; }
hr()  { printf '%s\n' "------------------------------------------------------------------"; }

if [ "$DRY_RUN" != "1" ] && ! command -v "$NVCC" >/dev/null 2>&1; then
    say "error: $NVCC not found. Load your CUDA module, or set NVCC=/path/to/nvcc."
    say "       (DRY_RUN=1 will print the plan without compiling anything.)"
    exit 1
fi

# Rung 4 must be enabled or every config reports "NOT IMPLEMENTED" and the heatmap is empty.
if grep -qE '^\s*static constexpr bool blocktile_2d\s*=\s*false' src/sgemm/kernels.cuh; then
    say ""
    say "*** LadderStatus::blocktile_2d is false in src/sgemm/kernels.cuh. ***"
    say "    04_2d_blocktile is disabled, so every config in this sweep would"
    say "    be reported as skipped and the heatmap would be empty."
    say "    Enable it in kernels.cuh before sweeping."
    if [ "${FORCE:-0}" != "1" ]; then
        say "    (FORCE=1 to run the sweep anyway.)"
        exit 2
    fi
fi

# One JSONL line per config; bench/sweep.py switches on sweep_status (ok|skipped|compile_error|run_error|incorrect).
emit_failure() {
    local status="$1" reason="$2" bm="$3" bn="$4" bk="$5" tm="$6" tn="$7" thr="$8" smem="$9"
    local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # Escape quotes/backslashes so nvcc error text cannot corrupt the JSONL.
    local esc; esc="$(printf '%s' "$reason" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n\t' '  ')"
    printf '{"timestamp":"%s","kernel":"04_2d_blocktile","config":"%s",' \
           "$ts" "BM${bm}_BN${bn}_BK${bk}_TM${tm}_TN${tn}" >> "$OUT"
    printf '"BM":%d,"BN":%d,"BK":%d,"TM":%d,"TN":%d,' "$bm" "$bn" "$bk" "$tm" "$tn" >> "$OUT"
    printf '"threads":%d,"smem_bytes":%d,"M":%d,"N":%d,"K":%d,' \
           "$thr" "$smem" "$SIZE" "$SIZE" "$SIZE" >> "$OUT"
    printf '"sweep_status":"%s","reason":"%s"}\n' "$status" "$esc" >> "$OUT"
}

say ""
say "== tile sweep for 04_2d_blocktile =="
say "   grid      BM={$BM_LIST} BN={$BN_LIST} BK={$BK_LIST} TM={$TM_LIST} TN={$TN_LIST}"
say "   timed at  ${SIZE}x${SIZE}x${SIZE}, warmup=$WARMUP iters=$ITERS"
say "   verify at ${VERIFY_SIZE:-<disabled>}"
say "   arch      $ARCH"
say "   output    $OUT"
say ""

n_total=0 n_ok=0 n_skip=0 n_fail=0

for BM in $BM_LIST; do
for BN in $BN_LIST; do
for BK in $BK_LIST; do
for TM in $TM_LIST; do
for TN in $TN_LIST; do
    n_total=$((n_total + 1))
    CFG="BM${BM}_BN${BN}_BK${BK}_TM${TM}_TN${TN}"

    # Reject impossible configs with cheap arithmetic before invoking nvcc.
    THREADS=$(( (BM * BN) / (TM * TN) ))
    SMEM=$(( (BM * BK + BK * BN) * 4 ))

    reason=""
    if [ $((BM % TM)) -ne 0 ]; then
        reason="invalid geometry: BM=$BM not divisible by TM=$TM"
    elif [ $((BN % TN)) -ne 0 ]; then
        reason="invalid geometry: BN=$BN not divisible by TN=$TN"
    elif [ "$THREADS" -le 0 ]; then
        reason="invalid geometry: (BM*BN)/(TM*TN) is not positive"
    elif [ $((THREADS % 32)) -ne 0 ]; then
        reason="invalid geometry: $THREADS threads is not a multiple of the 32-wide warp"
    elif [ "$THREADS" -gt "$MAX_THREADS" ]; then
        reason="$THREADS threads/block exceeds the $MAX_THREADS cap (raise TM/TN or lower BM/BN)"
    elif [ "$SMEM" -gt "$MAX_SMEM_BYTES" ]; then
        reason="needs $((SMEM / 1024)) KB shared memory, GB10 caps a block at $((MAX_SMEM_BYTES / 1024)) KB"
    elif [ $(( (BM * BK) % THREADS )) -ne 0 ]; then
        reason="A-tile (BM*BK=$((BM * BK))) does not divide evenly among $THREADS threads"
    elif [ $(( (BK * BN) % THREADS )) -ne 0 ]; then
        reason="B-tile (BK*BN=$((BK * BN))) does not divide evenly among $THREADS threads"
    fi

    if [ -n "$reason" ]; then
        n_skip=$((n_skip + 1))
        printf '%-34s SKIP  %s\n' "$CFG" "$reason"
        [ "$DRY_RUN" = "1" ] || emit_failure "skipped" "$reason" \
            "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM"
        continue
    fi

    printf '%-34s %4d thr, %3d KB smem ... ' "$CFG" "$THREADS" "$((SMEM / 1024))"

    if [ "$DRY_RUN" = "1" ]; then
        printf 'would build\n'
        continue
    fi

    BIN="$BUILD_DIR/ladder_$CFG"
    LOG="$BUILD_DIR/$CFG.build.log"

    # sm_121 needs CUDA >= 12.9; --threads 0 parallelizes nvcc since compile time dominates the sweep.
    "$NVCC" -O3 -std=c++17 -arch="$ARCH" --threads 0 \
        -DBM="$BM" -DBN="$BN" -DBK="$BK" -DTM="$TM" -DTN="$TN" \
        -Xptxas -v \
        src/sgemm/*.cu -lcublas -o "$BIN" > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
        n_fail=$((n_fail + 1))
        # Prefer the static_assert message over the first generic error line.
        msg="$(grep -m1 -E 'static assertion|static_assert|error:' "$LOG" | head -c 300)"
        [ -z "$msg" ] && msg="nvcc failed, see $LOG"
        printf 'COMPILE ERROR\n      %s\n' "$msg"
        emit_failure "compile_error" "$msg" "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM"
        continue
    fi

    # Capture ptxas register/spill counts so the heatmap can explain cold spots.
    SPILL="$(grep -oE '[0-9]+ bytes spill stores' "$LOG" | head -1 | grep -oE '^[0-9]+')"
    SPILL="${SPILL:-0}"
    REGS="$(grep -oE 'Used [0-9]+ registers' "$LOG" | head -1 | grep -oE '[0-9]+')"
    REGS="${REGS:-0}"

    VERIFIED="false"
    if [ -n "$VERIFY_SIZE" ]; then
        VOUT="$BUILD_DIR/$CFG.verify.jsonl"; rm -f "$VOUT"
        "$BIN" --only 04_2d_blocktile --size "$VERIFY_SIZE" \
               --warmup 2 --iters 3 --out "$VOUT" > "$BUILD_DIR/$CFG.verify.log" 2>&1
        if [ $? -ne 0 ] || [ ! -s "$VOUT" ]; then
            n_fail=$((n_fail + 1))
            msg="$(tail -3 "$BUILD_DIR/$CFG.verify.log" | head -c 300)"
            printf 'RUN ERROR (verify)\n'
            emit_failure "run_error" "verify run failed: $msg" \
                "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM"
            continue
        fi
        if grep -q '"correct":true' "$VOUT"; then
            VERIFIED="true"
        else
            n_fail=$((n_fail + 1))
            err="$(grep -oE '"max_rel_error":[^,]*' "$VOUT" | head -1)"
            printf 'INCORRECT at %s (%s)\n' "$VERIFY_SIZE" "$err"
            emit_failure "incorrect" "disagrees with CPU oracle at ${VERIFY_SIZE}: $err" \
                "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM"
            continue
        fi
    fi

    # --no-verify is safe only because the verify run above proved this exact binary correct.
    RAW="$BUILD_DIR/$CFG.raw.jsonl"; rm -f "$RAW"
    "$BIN" --only 04_2d_blocktile --size "$SIZE" \
           --warmup "$WARMUP" --iters "$ITERS" --no-verify --out "$RAW" \
           > "$BUILD_DIR/$CFG.run.log" 2>&1
    if [ $? -ne 0 ] || [ ! -s "$RAW" ]; then
        n_fail=$((n_fail + 1))
        msg="$(tail -3 "$BUILD_DIR/$CFG.run.log" | head -c 300)"
        printf 'RUN ERROR\n'
        emit_failure "run_error" "$msg" "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM"
        continue
    fi

    # Take the harness's JSONL record verbatim and add sweep fields, so bench/report.py can still read it.
    python3 - "$RAW" "$OUT" "$BM" "$BN" "$BK" "$TM" "$TN" "$THREADS" "$SMEM" \
             "$VERIFIED" "$VERIFY_SIZE" "$REGS" "$SPILL" <<'PY'
import json, sys
raw, out = sys.argv[1], sys.argv[2]
bm, bn, bk, tm, tn, thr, smem = (int(x) for x in sys.argv[3:10])
verified, vsize, regs, spill = sys.argv[10], sys.argv[11], int(sys.argv[12]), int(sys.argv[13])
lines = [l for l in open(raw).read().splitlines() if l.strip()]
if not lines:
    sys.exit(0)
rec = json.loads(lines[-1])          # last line = this config's run
rec.update({
    "config": f"BM{bm}_BN{bn}_BK{bk}_TM{tm}_TN{tn}",
    "BM": bm, "BN": bn, "BK": bk, "TM": tm, "TN": tn,
    "threads": thr, "smem_bytes": smem,
    "registers_per_thread": regs, "spill_bytes": spill,
    # The timed run used --no-verify; these fields record where correctness was actually established.
    "verified": verified == "true",
    "verified_at_size": vsize,
    "sweep_status": "skipped" if rec.get("skipped") else "ok",
})
with open(out, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
PY

    gf="$(grep -oE '"gflops_p50":[0-9.]+' "$RAW" | head -1 | cut -d: -f2)"
    n_ok=$((n_ok + 1))
    printf 'ok  %s GFLOP/s  (%s regs, %s B spill)\n' "${gf:-?}" "$REGS" "$SPILL"
done; done; done; done; done

hr
say "$n_total configs: $n_ok ran, $n_skip skipped before compiling, $n_fail failed"
say "results appended to $OUT"
say ""
say "next:  python3 bench/sweep.py $OUT"
say ""
say "Read the heatmap for the SHAPE, not just the maximum. A single hot cell"
say "surrounded by cold ones is usually measurement noise; a hot ridge tells you"
say "which parameter actually matters on this machine."
