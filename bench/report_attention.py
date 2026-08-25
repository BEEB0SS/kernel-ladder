#!/usr/bin/env python3
"""Report generator for the Phase-2 attention ladder."""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import report as R  # sibling module: palette, style, loaders, GB10 constants

# Keep ATTN_RUNGS in sync with src/attention/registry.cu.
ATTN_RUNGS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("00_naive_attention",  dict(precision="fp32", short="naive (3 kernels)")),
    ("01_fused_softmax",    dict(precision="fp32", short="fused softmax")),
    ("02_flash_tiled",      dict(precision="fp32", short="flash (online)")),
    ("03_flash_tensorcore", dict(precision="bf16", short="flash + mma")),
])
RUNG_INDEX = {name: i for i, name in enumerate(ATTN_RUNGS)}

RUNG_COLOR = {
    "00_naive_attention":  R.OKABE_ITO["grey"],
    "01_fused_softmax":    R.OKABE_ITO["sky"],
    "02_flash_tiled":      R.OKABE_ITO["blue"],
    "03_flash_tensorcore": R.OKABE_ITO["vermillion"],
}

DEFAULT_JSONL = "bench/results/attention.jsonl"


# FLOP/byte models mirror src/attention/attention_reference.hpp exactly.
def attention_flops(B: int, H: int, S: int, D: int, causal: bool) -> float:
    pairs = S * (S + 1) / 2.0 if causal else float(S) * S
    return 4.0 * B * H * pairs * D


def ideal_bytes(B: int, H: int, S: int, D: int, dtype_bytes: int) -> float:
    return 4.0 * B * H * S * D * dtype_bytes


def naive_bytes(B: int, H: int, S: int, D: int, dtype_bytes: int) -> float:
    # Q/K/V/O once, plus four full S x S passes through the score matrix.
    return (4.0 * S * D + 4.0 * float(S) * S) * B * H * dtype_bytes


def fused_bytes(B: int, H: int, S: int, D: int, dtype_bytes: int) -> float:
    # Rung 1: the softmax fusion removes two of the four S x S passes.
    return (4.0 * S * D + 2.0 * float(S) * S) * B * H * dtype_bytes


def rung_ai(kernel: str, B: int, H: int, S: int, D: int, causal: bool) -> float:
    """Arithmetic intensity under each rung's own traffic model."""
    b = 2 if ATTN_RUNGS.get(kernel, {}).get("precision") == "bf16" else 4
    fl = attention_flops(B, H, S, D, causal)
    if kernel == "00_naive_attention":
        return fl / naive_bytes(B, H, S, D, b)
    if kernel == "01_fused_softmax":
        return fl / fused_bytes(B, H, S, D, b)
    return fl / ideal_bytes(B, H, S, D, b)


def roof_gflops(precision: str, ai: float) -> float:
    peak = (R.FP32_PEAK_TFLOPS if precision == "fp32"
            else R.BF16_TENSOR_TFLOPS) * 1000.0
    return min(peak, R.DRAM_BW_MEASURED_GBS * ai)


def attention_rows(path: str) -> List[Dict[str, Any]]:
    rows = [r for r in R.load_jsonl(path) if r.get("phase") == "attention"]
    if not rows:
        sys.exit(f"error: {path} holds no attention records "
                 f"(phase != 'attention'; sgemm results go to report.py)")
    return rows


def family_key(r: Dict[str, Any]) -> Tuple[int, int, int]:
    return (r.get("B") or 0, r.get("H") or 0, r.get("D") or 0)


def latest(rows: Sequence[Dict[str, Any]], S: int, causal: bool
           ) -> Dict[str, Dict[str, Any]]:
    """Most recent record per kernel at one (S, causal); line breaks timestamp ties."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("S") != S or bool(r.get("causal")) != causal:
            continue
        if r.get("skipped") or not r.get("correct"):
            continue
        name = r.get("kernel") or ""
        key = (str(r.get("timestamp") or ""), r.get("_line", 0))
        prev = best.get(name)
        if prev is None or key > (str(prev.get("timestamp") or ""),
                                  prev.get("_line", 0)):
            best[name] = r
    return best


def print_table(rows: Sequence[Dict[str, Any]], B: int, H: int, D: int,
                S_list: Sequence[int], causal: bool) -> None:
    mode = "causal" if causal else "full"
    for S in S_list:
        recs = latest(rows, S, causal)
        if not recs:
            continue
        print(f"\n== attention {B}x{H}x{S}x{D}  ({mode}) ==")
        header = (f"{'kernel':<22}{'prec':<6}{'p50 ms':>9}{'GFLOP/s':>10}"
                  f"{'%roof':>7}{'vs prev':>9}{'eff GB/s':>10}{'cv':>7}"
                  f"{'max rel err':>13}")
        print(header)
        print("-" * len(header))
        prev: Optional[Dict[str, Any]] = None
        for name, meta in ATTN_RUNGS.items():
            r = recs.get(name)
            if r is None:
                print(f"{name:<22}{'—':<6}{'—':>9}{'—':>10}{'—':>7}"
                      f"{'—':>9}{'—':>10}{'—':>7}{'—':>13}")
                continue
            vs_prev = ""
            if prev and prev.get("p50_ms") and r.get("p50_ms"):
                vs_prev = f"{prev['p50_ms'] / r['p50_ms']:.2f}x"
            print(f"{name:<22}{meta['precision']:<6}"
                  f"{r.get('p50_ms', 0):>9.3f}{r.get('gflops_p50', 0):>10.1f}"
                  f"{r.get('pct_of_roofline', 0):>6.1f}%{vs_prev:>9}"
                  f"{r.get('effective_gbs', 0):>10.1f}{r.get('cv', 0):>7.3f}"
                  f"{r.get('max_rel_error', 0):>13.3g}")
            prev = r


def chart_ladder(rows: Sequence[Dict[str, Any]], B: int, H: int, D: int,
                 S: int, outdir: str, synthetic: bool) -> Optional[str]:
    plt = R._style()
    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals, colors, hatches = [], [], [], []
    for causal in (False, True):
        recs = latest(rows, S, causal)
        for name, meta in ATTN_RUNGS.items():
            r = recs.get(name)
            if r is None:
                continue
            names.append(f"{meta['short']}\n{'causal' if causal else 'full'}")
            vals.append(r.get("gflops_p50") or 0.0)
            colors.append(RUNG_COLOR[name])
            hatches.append("//" if meta["precision"] == "bf16" else "")
    if not vals:
        plt.close(fig)
        return None
    bars = ax.bar(range(len(vals)), vals, color=colors)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylabel("GFLOP/s (p50)")
    ax.set_yscale("log")
    ax.set_title(f"Attention ladder — {B}x{H}x{S}x{D} "
                 f"(hatched = bf16: fewer bits per number, not free speed)")
    if synthetic:
        R._mark_synthetic(fig, rows)
    path = os.path.join(outdir, "attention_ladder.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_scurve(rows: Sequence[Dict[str, Any]], B: int, H: int, D: int,
                 S_list: Sequence[int], outdir: str, synthetic: bool
                 ) -> Optional[str]:
    """GFLOP/s vs S against both roofs; predicted crossovers at S~514 fp32, S~1843 bf16."""
    plt = R._style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, meta in ATTN_RUNGS.items():
        xs, ys = [], []
        for S in S_list:
            r = latest(rows, S, False).get(name)
            if r and r.get("gflops_p50"):
                xs.append(S)
                ys.append(r["gflops_p50"])
        if xs:
            ax.plot(xs, ys, "o-", color=RUNG_COLOR[name], label=meta["short"])

    S_grid = sorted(set(S_list))
    for prec, style, label in (("fp32", "--", "fp32 achievable (S/4 AI)"),
                               ("bf16", ":", "bf16 achievable (S/2 AI)")):
        b = 4 if prec == "fp32" else 2
        roof = [roof_gflops(prec, S / b) for S in S_grid]
        ax.plot(S_grid, roof, style, color=R.OKABE_ITO["black"],
                linewidth=1.2, label=label)

    for x, prec in ((514, "fp32"), (1843, "bf16")):
        if S_grid and S_grid[0] <= x <= S_grid[-1]:
            ax.axvline(x, color=R.OKABE_ITO["yellow"], linewidth=1.0, zorder=0)
            ax.annotate(f"predicted {prec}\ncrossover S={x}", (x, ax.get_ylim()[0]),
                        xytext=(4, 12), textcoords="offset points", fontsize=7)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(S_grid, [str(s) for s in S_grid])
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("GFLOP/s (p50)")
    ax.set_title(f"Attention throughput vs S — B={B} H={H} D={D} (non-causal)")
    ax.legend(fontsize=8)
    if synthetic:
        R._mark_synthetic(fig, rows)
    path = os.path.join(outdir, "attention_scurve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_roofline(rows: Sequence[Dict[str, Any]], B: int, H: int, D: int,
                   S_list: Sequence[int], outdir: str, synthetic: bool
                   ) -> Optional[str]:
    plt = R._style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ai_grid = np.logspace(-1, 3.6, 200)
    for prec, style in (("fp32", "--"), ("bf16", ":")):
        ax.plot(ai_grid, [roof_gflops(prec, a) for a in ai_grid], style,
                color=R.OKABE_ITO["black"], linewidth=1.2,
                label=f"{prec} roof (231 GB/s / "
                      f"{R.FP32_PEAK_TFLOPS if prec == 'fp32' else R.BF16_TENSOR_TFLOPS:.1f} TF)")

    for name, meta in ATTN_RUNGS.items():
        xs, ys = [], []
        for S in S_list:
            r = latest(rows, S, False).get(name)
            if r and r.get("gflops_p50"):
                xs.append(rung_ai(name, B, H, S, D, False))
                ys.append(r["gflops_p50"])
        if xs:
            ax.plot(xs, ys, "o", color=RUNG_COLOR[name], label=meta["short"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte, per-rung traffic model)")
    ax.set_ylabel("GFLOP/s (p50)")
    ax.set_title(f"Attention roofline — B={B} H={H} D={D}, S swept "
                 f"{min(S_list)}..{max(S_list)}")
    ax.legend(fontsize=8)
    if synthetic:
        R._mark_synthetic(fig, rows)
    path = os.path.join(outdir, "attention_roofline.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def chart_latency(rows: Sequence[Dict[str, Any]], B: int, H: int, D: int,
                  S_list: Sequence[int], outdir: str, synthetic: bool
                  ) -> Optional[str]:
    plt = R._style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, meta in ATTN_RUNGS.items():
        xs, ys = [], []
        for S in S_list:
            r = latest(rows, S, False).get(name)
            if r and r.get("p50_ms"):
                xs.append(S)
                ys.append(r["p50_ms"])
        if xs:
            ax.plot(xs, ys, "o-", color=RUNG_COLOR[name], label=meta["short"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(set(S_list)), [str(s) for s in sorted(set(S_list))])
    ax.set_xlabel("sequence length S")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title(f"Attention latency vs S — B={B} H={H} D={D} (non-causal)")
    ax.legend(fontsize=8)
    if synthetic:
        R._mark_synthetic(fig, rows)
    path = os.path.join(outdir, "attention_latency.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="?", default=DEFAULT_JSONL)
    ap.add_argument("--outdir", default=R.DEFAULT_OUTDIR)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    rows = attention_rows(args.jsonl)
    synthetic = R.any_synthetic(rows)
    if synthetic:
        print(R.SYNTHETIC_BANNER)

    families: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        families.setdefault(family_key(r), []).append(r)

    for (B, H, D), fam in sorted(families.items()):
        S_list = sorted({r.get("S") or 0 for r in fam})
        for causal in (False, True):
            if any(bool(r.get("causal")) == causal for r in fam):
                print_table(fam, B, H, D, S_list, causal)
        if args.no_charts:
            continue
        S_head = max(S_list)
        made = [
            chart_ladder(fam, B, H, D, S_head, args.outdir, synthetic),
            chart_scurve(fam, B, H, D, S_list, args.outdir, synthetic),
            chart_roofline(fam, B, H, D, S_list, args.outdir, synthetic),
            chart_latency(fam, B, H, D, S_list, args.outdir, synthetic),
        ]
        for p in made:
            if p:
                print(f"chart: {p}")
    if synthetic:
        print(R.SYNTHETIC_BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
