#!/usr/bin/env python3
"""Test fixture: writes FAKE (synthetic=true) benchmark JSONL so report.py and sweep.py can be tested without a GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Invented fraction-of-ceiling per rung; not measured on a Spark.
SYNTHETIC_SHAPE = [
    # name,             precision,   frac_of_ceiling, implemented, correct, cv
    ("cublas",          "fp32",      0.62,  True,  True,  0.021),
    ("00_naive",        "fp32",      0.011, True,  True,  0.034),
    ("01_coalesced",    "fp32",      0.074, True,  True,  0.028),
    ("02_smem_tiled",   "fp32",      0.155, True,  True,  0.026),
    ("03_1d_blocktile", "fp32",      0.312, True,  True,  0.024),
    ("04_2d_blocktile", "fp32",      0.588, True,  True,  0.061),   # deliberately NOISY
    ("05_vectorized",   "fp32",      0.0,   True,  False, 0.0),     # deliberately WRONG
    ("cublas_tf32",     "bf16/tf32", 0.148, True,  True,  0.030),
    ("06_tensorcore",   "bf16/tf32", 0.0,   False, False, 0.0),     # deliberately SKIPPED
]

DESCRIPTIONS = {
    "cublas": "vendor library, strict fp32 (the number to beat)",
    "00_naive": "1 thread per output; threadIdx.x indexes ROW (uncoalesced)",
    "01_coalesced": "same math; threadIdx.x now indexes COLUMN",
    "02_smem_tiled": "stage BMxBK / BKxBN tiles in shared memory",
    "03_1d_blocktile": "TM outputs per thread; hoist the B load",
    "04_2d_blocktile": "TM x TN register tile per thread",
    "05_vectorized": "float4 loads + transposed A tile (no bank conflicts)",
    "cublas_tf32": "vendor library, TF32 tensor cores (baseline for rung 6)",
    "06_tensorcore": "bf16 mma.sync m16n8k16, fp32 accumulate",
}

FP32_PEAK_GFLOPS = 29.7 * 1000.0
BF16_PEAK_GFLOPS = 212.9 * 1000.0


def _percentiles(p50: float, rng: random.Random, cv: float) -> Dict[str, float]:
    """Right-skewed fake timings: min just below the median, long p99 tail."""
    jitter = max(cv, 0.005)
    return {
        "min_ms": p50 * (1.0 - jitter * 0.85),
        "p50_ms": p50,
        "p90_ms": p50 * (1.0 + jitter * 1.6),
        "p99_ms": p50 * (1.0 + jitter * 4.2 + rng.uniform(0.0, jitter)),
        "mean_ms": p50 * (1.0 + jitter * 0.45),
        "stddev_ms": p50 * jitter,
    }


def make_ladder(M: int, N: int, K: int, runs: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    flops = 2.0 * M * N * K
    out: List[Dict[str, Any]] = []
    t0 = datetime(2026, 8, 18, 9, 0, 0, tzinfo=timezone.utc)

    for run in range(runs):
        ts = (t0 + timedelta(days=run, minutes=rng.randint(0, 200)))
        stamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        # Later runs improve slightly so --history has something to show.
        improve = 1.0 + 0.06 * run

        for (name, prec, frac, implemented, correct, cv) in SYNTHETIC_SHAPE:
            rec: Dict[str, Any] = {
                "timestamp": stamp,
                "kernel": name,
                "description": DESCRIPTIONS[name],
                "M": M, "N": N, "K": K,
                "precision": prec,
                "skipped": not implemented,
                "correct": bool(correct),
                "max_rel_error": 0.0,
            }
            if not implemented:
                # Skipped rung: harness returns before timing, so fields stay zero.
                rec.update({k: 0.0 for k in
                            ("min_ms", "p50_ms", "p90_ms", "p99_ms",
                             "mean_ms", "stddev_ms", "cv", "gflops_p50",
                             "gflops_min", "pct_of_peak", "speedup_vs_cublas")})
                rec["n"] = 0
                rec.update({"sm_clock_before_mhz": 0.0, "sm_clock_after_mhz": 0.0,
                            "power_w_after": 0.0, "temp_c_after": 0.0,
                            "synthetic": True})
                out.append(rec)
                continue

            if not correct:
                # Wrong kernel: harness bails before timing; only max_rel_error is meaningful.
                rec["max_rel_error"] = 0.418
                rec.update({k: 0.0 for k in
                            ("min_ms", "p50_ms", "p90_ms", "p99_ms",
                             "mean_ms", "stddev_ms", "cv", "gflops_p50",
                             "gflops_min", "pct_of_peak", "speedup_vs_cublas")})
                rec["n"] = 0
                rec.update({"sm_clock_before_mhz": 2200.0, "sm_clock_after_mhz": 2180.0,
                            "power_w_after": 71.0, "temp_c_after": 58.0,
                            "synthetic": True})
                out.append(rec)
                continue

            ceiling = FP32_PEAK_GFLOPS if prec == "fp32" else BF16_PEAK_GFLOPS
            eff = frac * (improve if name.startswith(("03", "04", "05")) else 1.0)
            eff *= rng.uniform(0.985, 1.015)
            gf = ceiling * eff
            p50 = (flops / (gf * 1e9)) * 1e3
            pcts = _percentiles(p50, rng, cv)
            rec.update(pcts)
            rec["cv"] = cv
            rec["n"] = 100
            rec["gflops_p50"] = gf
            rec["gflops_min"] = flops / (pcts["min_ms"] * 1e-3) / 1e9
            rec["pct_of_peak"] = 100.0 * gf / ceiling
            rec["speedup_vs_cublas"] = 0.0     # report.py recomputes this anyway
            rec["sm_clock_before_mhz"] = round(rng.uniform(2150, 2400), 1)
            rec["sm_clock_after_mhz"] = round(rng.uniform(2050, 2350), 1)
            rec["power_w_after"] = round(rng.uniform(58, 88), 1)
            rec["temp_c_after"] = round(rng.uniform(52, 71), 1)
            rec["max_rel_error"] = round(rng.uniform(1e-6, 8e-5), 9)
            rec["synthetic"] = True
            out.append(rec)
    return out


# Keep in sync with what scripts/sweep_tiles.sh writes, including failure records.
SWEEP_BM = [64, 128]
SWEEP_BN = [64, 128]
SWEEP_BK = [8, 16]
SWEEP_TN = [4, 8]
SWEEP_TM = [4, 8]


def make_sweep(M: int, N: int, K: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 1)
    flops = 2.0 * M * N * K
    out: List[Dict[str, Any]] = []
    stamp = datetime(2026, 8, 20, 14, 30, 0, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    for BM in SWEEP_BM:
        for BN in SWEEP_BN:
            for BK in SWEEP_BK:
                for TM in SWEEP_TM:
                    for TN in SWEEP_TN:
                        cfg = f"BM{BM}_BN{BN}_BK{BK}_TM{TM}_TN{TN}"
                        threads = (BM * BN) // (TM * TN)
                        smem = (BM * BK + BK * BN) * 4
                        base = {
                            "timestamp": stamp, "kernel": "04_2d_blocktile",
                            "M": M, "N": N, "K": K, "precision": "fp32",
                            "config": cfg, "BM": BM, "BN": BN, "BK": BK,
                            "TM": TM, "TN": TN,
                            "threads": threads, "smem_bytes": smem,
                            "synthetic": True,
                        }
                        if threads > 1024:
                            out.append({**base, "sweep_status": "skipped",
                                        "reason": f"{threads} threads/block > 1024 cap"})
                            continue
                        if smem > 99 * 1024:
                            out.append({**base, "sweep_status": "skipped",
                                        "reason": f"{smem} B shared > 99KB GB10 cap"})
                            continue
                        # One invented compile failure and one runtime failure exercise sweep.py's error paths.
                        if (BM, BN, BK, TM, TN) == (64, 64, 16, 4, 4):
                            out.append({**base, "sweep_status": "compile_error",
                                        "reason": "static_assert: A-tile must divide "
                                                  "evenly among threads"})
                            continue
                        if (BM, BN, BK, TM, TN) == (128, 128, 16, 4, 4):
                            out.append({**base, "sweep_status": "run_error",
                                        "reason": "cudaErrorLaunchOutOfResources "
                                                  "(register pressure)"})
                            continue

                        # Fake performance surface with a peak near 128/128/8/8/8.
                        work = TM * TN
                        score = (1.0
                                 - 0.35 * abs(math.log2(work / 64.0)) / 3.0
                                 - 0.20 * abs(math.log2(BM / 128.0))
                                 - 0.20 * abs(math.log2(BN / 128.0))
                                 - 0.10 * abs(math.log2(BK / 8.0)))
                        eff = max(0.06, 0.62 * score) * rng.uniform(0.97, 1.03)
                        gf = FP32_PEAK_GFLOPS * eff
                        p50 = (flops / (gf * 1e9)) * 1e3
                        cv = round(rng.uniform(0.015, 0.05), 4)
                        pcts = _percentiles(p50, rng, cv)
                        out.append({**base, "sweep_status": "ok",
                                    "skipped": False, "correct": True,
                                    "max_rel_error": 3.1e-5,
                                    **pcts, "cv": cv, "n": 50,
                                    "gflops_p50": gf,
                                    "gflops_min": flops / (pcts["min_ms"] * 1e-3) / 1e9,
                                    "pct_of_peak": 100.0 * gf / FP32_PEAK_GFLOPS,
                                    "speedup_vs_cublas": 0.0,
                                    "sm_clock_before_mhz": round(rng.uniform(2150, 2400), 1),
                                    "sm_clock_after_mhz": round(rng.uniform(2050, 2350), 1),
                                    "power_w_after": round(rng.uniform(58, 88), 1),
                                    "temp_c_after": round(rng.uniform(52, 71), 1)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--sweep", action="store_true",
                    help="generate a tile-sweep fixture instead of a ladder fixture")
    ap.add_argument("--size", default="4096x4096x4096", help="MxNxK")
    ap.add_argument("--runs", type=int, default=3,
                    help="how many separate runs to fake (exercises --history)")
    ap.add_argument("--extra-size", default="1024x2048x512",
                    help="a second, non-square size; '' to skip. Non-square sizes "
                         "are how you catch M/N transposition bugs, so the fixture "
                         "has one.")
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    M, N, K = (int(x) for x in args.size.lower().split("x"))
    if args.sweep:
        recs = make_sweep(M, N, K, args.seed)
        out = args.out or "bench/results/synthetic_sweep.jsonl"
    else:
        recs = make_ladder(M, N, K, args.runs, args.seed)
        if args.extra_size:
            m2, n2, k2 = (int(x) for x in args.extra_size.lower().split("x"))
            recs += make_ladder(m2, n2, k2, 1, args.seed + 7)
        out = args.out or "bench/results/synthetic_sgemm.jsonl"

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {len(recs)} SYNTHETIC records to {out}")
    print("*** These are INVENTED numbers for testing the reporting code. ***")
    print("*** They are not measurements. Do not put them in a writeup.   ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
