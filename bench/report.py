#!/usr/bin/env python3
"""report.py — Turn bench/results/*.jsonl into a ladder table and the charts.

USAGE
-----
    python3 bench/report.py                                  # newest results, default file
    python3 bench/report.py bench/results/sgemm.jsonl
    python3 bench/report.py --size 4096x4096x4096            # pin one problem size
    python3 bench/report.py --markdown                       # paste into README
    python3 bench/report.py --history                        # every rung over time
    python3 bench/report.py --history 04_2d_blocktile        # one rung over time
    python3 bench/report.py --no-charts                      # table only, no matplotlib

Charts land in bench/results/ as PNG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Hardware constants — keep in sync with src/common/gb10.hpp (duplicated on purpose).
# Measured numbers where possible; roofline uses measured DRAM bandwidth, not spec.
DRAM_BW_MEASURED_GBS = 231.0     # saturating-read kernel, 8 GiB buffer
DRAM_BW_PEAK_GBS = 273.0         # spec sheet, 256-bit LPDDR5X @ 8533 MT/s
FP32_PEAK_TFLOPS = 29.7          # CUDA cores, derived: 6144 * 2 FLOP * clock
TF32_TENSOR_TFLOPS = 53.3        # measured (mmapeak)
BF16_TENSOR_TFLOPS = 212.9       # measured (mmapeak)
NUM_SMS = 48
MAX_SMEM_PER_BLOCK_KB = 99       # GB10 per-block smem cap

# Ridge point: arithmetic intensity where bandwidth-limited becomes ALU-limited.
RIDGE_FP32 = (FP32_PEAK_TFLOPS * 1e12) / (DRAM_BW_MEASURED_GBS * 1e9)   # ~129
RIDGE_BF16 = (BF16_TENSOR_TFLOPS * 1e12) / (DRAM_BW_MEASURED_GBS * 1e9)  # ~921

# Rung order — keep in sync with src/sgemm/registry.cu (ladder order, not alphabetical).
# baseline=True marks vendor libraries: shown in the table but skipped by "vs prev".
RUNGS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("cublas",         dict(baseline=True,  precision="fp32",      short="cuBLAS fp32")),
    ("00_naive",       dict(baseline=False, precision="fp32",      short="naive")),
    ("01_coalesced",   dict(baseline=False, precision="fp32",      short="coalesced")),
    ("02_smem_tiled",  dict(baseline=False, precision="fp32",      short="smem tiled")),
    ("03_1d_blocktile",dict(baseline=False, precision="fp32",      short="1D blocktile")),
    ("04_2d_blocktile",dict(baseline=False, precision="fp32",      short="2D blocktile")),
    ("05_vectorized",  dict(baseline=False, precision="fp32",      short="vectorized")),
    ("cublas_tf32",    dict(baseline=True,  precision="bf16/tf32", short="cuBLAS TF32")),
    ("06_tensorcore",  dict(baseline=False, precision="bf16/tf32", short="tensor core")),
])
RUNG_INDEX = {name: i for i, name in enumerate(RUNGS)}

# fp32 kernels compare against strict-fp32 cuBLAS; reduced precision against TF32 cuBLAS.
BASELINE_FOR = {"fp32": "cublas", "bf16/tf32": "cublas_tf32"}

# Okabe-Ito colorblind-safe palette; also survives greyscale printing.
OKABE_ITO = {
    "black":       "#000000",
    "orange":      "#E69F00",
    "sky":         "#56B4E9",
    "green":       "#009E73",
    "yellow":      "#F0E442",
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "purple":      "#CC79A7",
    "grey":        "#666666",
}
FP32_COLOR = OKABE_ITO["blue"]
TC_COLOR = OKABE_ITO["vermillion"]
BASELINE_COLOR = OKABE_ITO["grey"]
CEILING_COLOR = OKABE_ITO["black"]

DEFAULT_JSONL = "bench/results/sgemm.jsonl"
DEFAULT_OUTDIR = "bench/results"


# harness.cuh writes JSONL via fprintf, which can emit bare nan/inf (not legal JSON).
_NONFINITE = re.compile(r':\s*(-?)(?:nan|inf|NAN|INF)(?=\s*[,}])')


def _sanitize_json_line(line: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        sign = m.group(1)
        token = m.group(0)
        return ": null" if "nan" in token.lower() else f": {sign}Infinity"
    return _NONFINITE.sub(repl, line)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read the harness output. Returns one dict per benchmarked kernel."""
    if not os.path.exists(path):
        sys.exit(
            f"error: no results file at {path}\n"
            f"       run the benchmark first, e.g.:  ./build/ladder --size 4096\n"
            f"       (or pass a path: python3 bench/report.py path/to/results.jsonl)"
        )
    rows: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(_sanitize_json_line(raw), parse_constant=lambda _c: None)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 3:
                    print(f"warning: {path}:{lineno} is not valid JSON, skipping",
                          file=sys.stderr)
                continue
            rec["_line"] = lineno
            rows.append(rec)
    if bad > 3:
        print(f"warning: {bad} unparseable lines total in {path}", file=sys.stderr)
    if not rows:
        sys.exit(f"error: {path} contained no usable records")
    return rows


def any_synthetic(rows: Sequence[Dict[str, Any]]) -> bool:
    """True if any record was invented by bench/make_synthetic_results.py."""
    return any(bool(r.get("synthetic")) for r in rows)


SYNTHETIC_BANNER = (
    "!!! SYNTHETIC DATA — these numbers were INVENTED by "
    "bench/make_synthetic_results.py for testing this script. "
    "They are NOT measurements from a GPU. Do not quote them. !!!"
)


def size_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('M')}x{rec.get('N')}x{rec.get('K')}"


def sizes_in(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Problem sizes present, biggest first (biggest is usually the headline)."""
    seen = {}
    for r in rows:
        k = size_key(r)
        seen[k] = max(seen.get(k, 0), (r.get("M") or 0) * (r.get("N") or 0) * (r.get("K") or 0))
    return sorted(seen, key=lambda k: -seen[k])


def latest_per_kernel(rows: Sequence[Dict[str, Any]], size: str) -> Dict[str, Dict[str, Any]]:
    """Most recent record per kernel; line number breaks whole-second timestamp ties."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if size_key(r) != size:
            continue
        name = r.get("kernel")
        if not name:
            continue
        key = (str(r.get("timestamp") or ""), r.get("_line", 0))
        prev = best.get(name)
        if prev is None or key > (str(prev.get("timestamp") or ""), prev.get("_line", 0)):
            best[name] = r
    return best


def gemm_flops(M: int, N: int, K: int) -> float:
    """2*M*N*K. Same convention as cpu_reference.hpp: alpha/beta work ignored."""
    return 2.0 * M * N * K


def compulsory_bytes(M: int, N: int, K: int, precision: str) -> float:
    """Minimum DRAM traffic: A and B read once (2 bytes if reduced precision), C written as fp32."""
    ab_bytes = 2 if precision != "fp32" else 4
    return (M * K + K * N) * ab_bytes + (M * N) * 4


def arithmetic_intensity(rec: Dict[str, Any]) -> Tuple[float, bool]:
    """(FLOP per byte, measured?) — uses a logged dram_bytes field, else compulsory traffic."""
    M, N, K = rec.get("M", 0), rec.get("N", 0), rec.get("K", 0)
    flops = gemm_flops(M, N, K)
    measured = rec.get("dram_bytes")
    if isinstance(measured, (int, float)) and measured > 0:
        return flops / float(measured), True
    return flops / compulsory_bytes(M, N, K, rec.get("precision", "fp32")), False


def ceiling_gflops(precision: str) -> float:
    """Compute ceiling — mirrors harness.cuh so pct_of_peak and '% ceil' never disagree."""
    return (FP32_PEAK_TFLOPS if precision == "fp32" else BF16_TENSOR_TFLOPS) * 1000.0


def status_of(rec: Dict[str, Any]) -> str:
    """ok | skipped | wrong — skipped first: a skipped kernel's `correct` is uninitialized."""
    if rec.get("skipped"):
        return "skipped"
    if not rec.get("correct", False):
        return "wrong"
    return "ok"


def is_usable(rec: Optional[Dict[str, Any]]) -> bool:
    """Usable = it produced a number we are allowed to quote."""
    return (rec is not None
            and status_of(rec) == "ok"
            and isinstance(rec.get("gflops_p50"), (int, float))
            and rec["gflops_p50"] > 0)


class Row:
    """One line of the report."""

    __slots__ = ("name", "precision", "status", "rec", "p50_ms", "gflops",
                 "pct_ceiling", "vs_prev", "vs_prev_name", "vs_prev_is_adjacent",
                 "vs_baseline", "baseline_name", "cv", "is_baseline", "note")

    def __init__(self, name: str) -> None:
        self.name = name
        self.precision = RUNGS.get(name, {}).get("precision", "?")
        self.status = "missing"
        self.rec: Optional[Dict[str, Any]] = None
        self.p50_ms = float("nan")
        self.gflops = float("nan")
        self.pct_ceiling = float("nan")
        self.vs_prev: Optional[float] = None
        self.vs_prev_name: Optional[str] = None
        self.vs_prev_is_adjacent = True
        self.vs_baseline: Optional[float] = None
        self.baseline_name: Optional[str] = None
        self.cv = float("nan")
        self.is_baseline = bool(RUNGS.get(name, {}).get("baseline"))
        self.note = ""


def build_rows(latest: Dict[str, Dict[str, Any]]) -> List[Row]:
    """Assemble the table in REGISTRY order, filling in both speedup columns."""
    # Kernels in the file but not in RUNGS are appended so they do not vanish.
    names = list(RUNGS.keys())
    names += sorted(n for n in latest if n not in RUNG_INDEX)

    rows: List[Row] = []
    for name in names:
        rec = latest.get(name)
        row = Row(name)
        if rec is None:
            row.status = "missing"
            rows.append(row)
            continue
        row.rec = rec
        row.precision = rec.get("precision") or row.precision
        row.status = status_of(rec)
        if row.status == "ok":
            row.p50_ms = float(rec.get("p50_ms") or float("nan"))
            row.gflops = float(rec.get("gflops_p50") or float("nan"))
            row.cv = float(rec.get("cv") or float("nan"))
            ceil = ceiling_gflops(row.precision)
            row.pct_ceiling = 100.0 * row.gflops / ceil if ceil > 0 else float("nan")
        rows.append(row)

    # Recomputed from the file (not the JSONL's speedup_vs_cublas) so the report
    # is order-independent even for --only runs.
    for row in rows:
        base_name = BASELINE_FOR.get(row.precision)
        base = latest.get(base_name) if base_name else None
        if row.status == "ok" and is_usable(base):
            row.vs_baseline = row.gflops / float(base["gflops_p50"])
            row.baseline_name = base_name

    # Previous rung = nearest earlier non-baseline with a result; a non-adjacent
    # fallback is flagged via vs_prev_is_adjacent (not a single-variable comparison).
    ladder_rows = [r for r in rows if not r.is_baseline and r.name in RUNG_INDEX]
    ladder_rows.sort(key=lambda r: RUNG_INDEX[r.name])
    for i, row in enumerate(ladder_rows):
        if row.status != "ok":
            continue
        for j in range(i - 1, -1, -1):
            prev = ladder_rows[j]
            if prev.status == "ok" and prev.gflops > 0:
                row.vs_prev = row.gflops / prev.gflops
                row.vs_prev_name = prev.name
                row.vs_prev_is_adjacent = (j == i - 1)
                break

    for row in rows:
        bits = []
        if row.status == "wrong" and row.rec is not None:
            err = row.rec.get("max_rel_error")
            bits.append("WRONG (max rel err %s)" %
                        ("nan/inf" if err is None else f"{err:.3g}"))
        elif row.status == "skipped":
            bits.append("not implemented yet")
        elif row.status == "missing":
            bits.append("no result at this size")
        else:
            # cv threshold matches print_result() in harness.cuh.
            if not math.isnan(row.cv) and row.cv > 0.05:
                bits.append("NOISY cv>5%")
            rec = row.rec or {}
            before = rec.get("sm_clock_before_mhz") or 0.0
            after = rec.get("sm_clock_after_mhz") or 0.0
            # GB10 cannot lock clocks; >10% drift during the run invalidates the number.
            if before > 0 and after > 0 and abs(after - before) / before > 0.10:
                bits.append(f"clock drift {before:.0f}->{after:.0f} MHz")
            if 0 < after < 900:
                bits.append(f"SM clock {after:.0f} MHz — GB10 stuck-clock? power cycle")
        row.note = "; ".join(bits)
    return rows


def _fmt(v: Optional[float], spec: str, dash: str = "—") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return dash
    return format(v, spec)


def print_table(rows: List[Row], size: str, path: str,
                synthetic: bool = False) -> None:
    if synthetic:
        bar = "*" * len(SYNTHETIC_BANNER)
        print(f"\n{bar}\n{SYNTHETIC_BANNER}\n{bar}")
    hdr = (f"{'kernel':<18}{'prec':<11}{'p50 ms':>9}{'GFLOP/s':>10}"
           f"{'% ceil':>8}{'vs prev':>9}{'vs cuBLAS':>11}{'cv':>7}  notes")
    print()
    print(f"== SGEMM ladder =={'':<2}{size}{'':<2}(source: {path})")
    print(hdr)
    print("-" * max(len(hdr), 96))

    for row in rows:
        if row.status == "missing":
            print(f"{row.name:<18}{row.precision:<11}{'—':>9}{'—':>10}"
                  f"{'—':>8}{'—':>9}{'—':>11}{'—':>7}  {row.note}")
            continue
        if row.status != "ok":
            print(f"{row.name:<18}{row.precision:<11}{'—':>9}{'—':>10}"
                  f"{'—':>8}{'—':>9}{'—':>11}{'—':>7}  {row.note}")
            continue

        vs_prev = "—" if row.vs_prev is None else (
            f"{row.vs_prev:.2f}x" + ("" if row.vs_prev_is_adjacent else "†"))
        vs_base = "—" if row.vs_baseline is None else f"{row.vs_baseline:.2f}x"
        if row.is_baseline and row.vs_baseline is not None and abs(row.vs_baseline - 1.0) < 1e-9:
            vs_base = "(base)"
        print(f"{row.name:<18}{row.precision:<11}"
              f"{_fmt(row.p50_ms, '9.3f')}"
              f"{_fmt(row.gflops, '10.1f')}"
              f"{_fmt(row.pct_ceiling, '7.1f')}%"
              f"{vs_prev:>9}{vs_base:>11}"
              f"{_fmt(row.cv, '7.3f')}  {row.note}")

    print()
    _print_footnotes(rows)


def _print_footnotes(rows: List[Row]) -> None:
    """Everything the columns cannot say without lying by omission."""
    notes: List[str] = []

    dagger = [r for r in rows
              if r.vs_prev is not None and not r.vs_prev_is_adjacent]
    if dagger:
        for r in dagger:
            notes.append(
                f"† {r.name}: 'vs prev' is against {r.vs_prev_name}, not the "
                f"adjacent rung (that rung has no result here). This bundles "
                f"more than one change into a single number — implement the "
                f"missing rung before quoting it as a single-variable result.")

    notes.append(
        f"'% ceil' is against the compute roofline the harness uses: "
        f"{FP32_PEAK_TFLOPS:g} TFLOP/s for fp32 (CUDA cores), "
        f"{BF16_TENSOR_TFLOPS:g} TFLOP/s for bf16/tf32 (tensor cores).")

    if any(r.name == "cublas_tf32" and r.status == "ok" for r in rows):
        tf32 = next(r for r in rows if r.name == "cublas_tf32")
        alt = 100.0 * tf32.gflops / (TF32_TENSOR_TFLOPS * 1000.0)
        notes.append(
            f"cublas_tf32 is scored against the BF16 ceiling "
            f"({BF16_TENSOR_TFLOPS:g} TFLOP/s) because that is the rule in "
            f"harness.cuh. Against the TF32 ceiling ({TF32_TENSOR_TFLOPS:g} "
            f"TFLOP/s) it is at {alt:.1f}%, which is the fairer reading.")

    notes.append(
        "'vs prev' skips the vendor baselines: it compares each rung to the "
        "rung below it, which is the only comparison where exactly one thing "
        "changed.")

    notes.append(
        "cv is stddev/mean of the per-iteration times. Above 0.05 the "
        "measurement is dirty (GB10 cannot lock clocks) — rerun on a quiet box "
        "before believing a small difference.")

    for n in notes:
        # Wrap by hand rather than importing textwrap for one call site.
        words, line = n.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 92:
                print("  " + line)
                line = "    " + w
            else:
                line = (line + " " + w) if line else w
        if line:
            print("  " + line)
    print()


def print_markdown(rows: List[Row], size: str, device_note: str = "",
                   synthetic: bool = False) -> None:
    """A markdown table suitable for pasting into a README."""
    print()
    if synthetic:
        print(f"> **{SYNTHETIC_BANNER}**")
        print()
    print(f"### SGEMM ladder — {size}")
    print()
    if device_note:
        print(device_note)
        print()
    print("| rung | precision | p50 (ms) | GFLOP/s | % of ceiling | vs prev rung | vs cuBLAS | cv |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.status == "missing":
            continue
        if row.status != "ok":
            reason = "not implemented" if row.status == "skipped" else "**INCORRECT**"
            print(f"| `{row.name}` | {row.precision} | — | — | — | — | — | — |"
                  .replace("| — | — | — | — | — | — |", f"| {reason} | — | — | — | — | — |", 1))
            continue
        vs_prev = "—" if row.vs_prev is None else (
            f"{row.vs_prev:.2f}x" + ("†" if not row.vs_prev_is_adjacent else ""))
        vs_base = "—" if row.vs_baseline is None else f"{row.vs_baseline:.2f}x"
        if row.is_baseline and row.vs_baseline is not None and abs(row.vs_baseline - 1.0) < 1e-9:
            vs_base = "_baseline_"
        print(f"| `{row.name}` | {row.precision} | {row.p50_ms:.3f} | "
              f"{row.gflops:.1f} | {row.pct_ceiling:.1f}% | {vs_prev} | "
              f"{vs_base} | {row.cv:.3f} |")
    print()
    print(f"Ceilings: fp32 {FP32_PEAK_TFLOPS:g} TFLOP/s (CUDA cores), "
          f"bf16 {BF16_TENSOR_TFLOPS:g} TFLOP/s (tensor cores), "
          f"DRAM {DRAM_BW_MEASURED_GBS:g} GB/s (measured, not the "
          f"{DRAM_BW_PEAK_GBS:g} GB/s spec figure).")
    print(f"Timings are CUDA-event p50 over the measured iterations, after "
          f"warmup, gated on a CPU-oracle correctness check. "
          f"GB10 cannot lock clocks, hence the cv column.")
    print()


def print_history(rows_all: Sequence[Dict[str, Any]], size: str,
                  only_kernel: Optional[str]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Every run of each kernel at one size, oldest first, with per-run deltas."""
    by_kernel: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for rec in rows_all:
        if size_key(rec) != size:
            continue
        name = rec.get("kernel")
        if not name or (only_kernel and name != only_kernel):
            continue
        by_kernel.setdefault(name, []).append(rec)

    ordered = sorted(by_kernel.items(),
                     key=lambda kv: RUNG_INDEX.get(kv[0], 10_000))
    if not ordered:
        target = only_kernel or "any kernel"
        print(f"\nno history for {target} at {size}\n")
        return []

    print()
    print(f"== history =={'':<2}{size}")
    for name, recs in ordered:
        recs.sort(key=lambda r: (str(r.get("timestamp") or ""), r.get("_line", 0)))
        print(f"\n  {name}")
        print(f"    {'timestamp':<22}{'p50 ms':>9}{'GFLOP/s':>10}{'cv':>7}"
              f"{'delta':>9}{'SM MHz':>9}  status")
        prev_g = None
        for rec in recs:
            st = status_of(rec)
            ts = str(rec.get("timestamp") or "?")
            if st != "ok":
                print(f"    {ts:<22}{'—':>9}{'—':>10}{'—':>7}{'—':>9}{'—':>9}  {st}")
                continue
            g = float(rec.get("gflops_p50") or 0.0)
            delta = "—" if not prev_g else f"{g / prev_g:+.1%}".replace("+-", "-")
            if prev_g:
                delta = f"{(g / prev_g - 1.0):+.1%}"
            clk = rec.get("sm_clock_after_mhz") or 0.0
            print(f"    {ts:<22}{float(rec.get('p50_ms') or 0):>9.3f}{g:>10.1f}"
                  f"{float(rec.get('cv') or 0):>7.3f}{delta:>9}"
                  f"{clk:>9.0f}  ok")
            prev_g = g
    print()
    print("  A delta smaller than the cv of either run is not a result, it is "
          "weather. On GB10 that is often several percent.")
    print()
    return ordered


def _style():
    import matplotlib
    matplotlib.use("Agg")   # no display on a headless Spark or a build box
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#DDDDDD",
        "legend.frameon": False,
    })
    return plt


def _mark_synthetic(fig, items) -> None:
    """Watermark a chart drawn from synthetic fixture data."""
    recs = []
    for it in items:
        rec = getattr(it, "rec", None) if not isinstance(it, dict) else it
        if rec:
            recs.append(rec)
    if not any(bool(r.get("synthetic")) for r in recs):
        return
    fig.text(0.5, 0.5, "SYNTHETIC\nTEST DATA", fontsize=54, color="#D55E00",
             alpha=0.16, ha="center", va="center", rotation=24,
             fontweight="bold", zorder=100)


def chart_ladder(rows: List[Row], size: str, outdir: str) -> Optional[str]:
    """GFLOP/s per rung, with cuBLAS and roofline reference lines."""
    plt = _style()
    usable = [r for r in rows if r.status == "ok" and not r.is_baseline]
    if not usable:
        print("skipping ladder chart: no correct non-baseline results", file=sys.stderr)
        return None
    usable.sort(key=lambda r: RUNG_INDEX.get(r.name, 10_000))

    labels = [f"{r.name}\n{RUNGS.get(r.name, {}).get('short', '')}" for r in usable]
    vals = [r.gflops for r in usable]
    colors = [TC_COLOR if r.precision != "fp32" else FP32_COLOR for r in usable]
    # Hatching marks reduced-precision bars: not the same arithmetic as fp32.
    hatches = ["//" if r.precision != "fp32" else "" for r in usable]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.35 * len(usable) + 3.0), 5.6))
    bars = ax.bar(range(len(usable)), vals, color=colors, width=0.68,
                  edgecolor="#222222", linewidth=0.8)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)

    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("GFLOP/s (from median iteration time)")
    ax.set_title(f"SGEMM optimization ladder — DGX Spark (GB10, sm_121) — {size}")
    _mark_synthetic(fig, rows)
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)

    for b, r in zip(bars, usable):
        ax.annotate(f"{r.gflops:,.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    xytext=(0, 2), textcoords="offset points")
        if r.vs_prev is not None:
            ax.annotate(f"{r.vs_prev:.2f}x vs prev" + ("†" if not r.vs_prev_is_adjacent else ""),
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color="#444444",
                        xytext=(0, 14), textcoords="offset points")

    ymax = max(vals)
    ref_lines = []

    cublas = next((r for r in rows if r.name == "cublas" and r.status == "ok"), None)
    if cublas:
        ax.axhline(cublas.gflops, color=BASELINE_COLOR, linestyle="--", linewidth=1.6)
        ref_lines.append((cublas.gflops, f"cuBLAS fp32  {cublas.gflops:,.0f} GFLOP/s"))
        ymax = max(ymax, cublas.gflops)

    tf32 = next((r for r in rows if r.name == "cublas_tf32" and r.status == "ok"), None)
    if tf32:
        ax.axhline(tf32.gflops, color=BASELINE_COLOR, linestyle=":", linewidth=1.6)
        ref_lines.append((tf32.gflops, f"cuBLAS TF32  {tf32.gflops:,.0f} GFLOP/s"))
        ymax = max(ymax, tf32.gflops)

    fp32_ceiling = FP32_PEAK_TFLOPS * 1000.0
    ax.axhline(fp32_ceiling, color=CEILING_COLOR, linestyle="-", linewidth=1.4)
    ref_lines.append((fp32_ceiling,
                      f"fp32 roofline  {fp32_ceiling:,.0f} GFLOP/s (CUDA cores)"))
    ymax = max(ymax, fp32_ceiling)

    # Draw the bf16 ceiling only when a bar is near it, else fp32 bars flatten into the floor.
    if any(r.precision != "fp32" for r in usable):
        bf16_ceiling = BF16_TENSOR_TFLOPS * 1000.0
        if max(vals) > 0.15 * bf16_ceiling:
            ax.axhline(bf16_ceiling, color=CEILING_COLOR, linestyle="-.", linewidth=1.2)
            ref_lines.append((bf16_ceiling,
                              f"bf16 tensor roofline  {bf16_ceiling:,.0f} GFLOP/s"))
            ymax = max(ymax, bf16_ceiling)

    ax.set_ylim(0, ymax * 1.18)
    for y, text in ref_lines:
        ax.annotate(text, (-0.45, y), ha="left", va="bottom",
                    fontsize=8.5, color="#222222",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="#CCCCCC", lw=0.6))

    fig.text(0.005, -0.02,
             "Bars are the ladder rungs, in registry order. Hatched = reduced "
             "precision (bf16 in, fp32 accumulate) — not the same arithmetic, so "
             "not a free win.\nEach rung changes exactly one thing versus the rung "
             "to its left; the small label is that change's speedup.",
             fontsize=8, color="#555555", va="top")

    path = os.path.join(outdir, "ladder.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_roofline(rows: List[Row], size: str, outdir: str) -> Optional[str]:
    """Log-log roofline: achieved GFLOP/s against the DRAM and compute ceilings."""
    plt = _style()
    import numpy as np

    pts = [r for r in rows if r.status == "ok" and r.rec is not None]
    if not pts:
        print("skipping roofline: no correct results", file=sys.stderr)
        return None

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    ais = [arithmetic_intensity(r.rec)[0] for r in pts]
    gfs = [r.gflops for r in pts]
    x_lo = min(0.5, min(ais) * 0.4)
    x_hi = max(RIDGE_BF16 * 2.2, max(ais) * 2.6)
    xs = np.logspace(math.log10(x_lo), math.log10(x_hi), 512)

    # y[GFLOP/s] = BW[GB/s] * AI[FLOP/byte]; the units cancel to GFLOP/s.
    mem_line = DRAM_BW_MEASURED_GBS * xs
    ax.plot(xs, mem_line, color=OKABE_ITO["sky"], linewidth=2.0,
            label=f"DRAM {DRAM_BW_MEASURED_GBS:g} GB/s (measured)")
    ax.plot(xs, DRAM_BW_PEAK_GBS * xs, color=OKABE_ITO["sky"], linewidth=1.0,
            linestyle=":", alpha=0.8,
            label=f"DRAM {DRAM_BW_PEAK_GBS:g} GB/s (spec sheet)")

    for tflops, color, style, name in [
        (FP32_PEAK_TFLOPS, FP32_COLOR, "-", "fp32 CUDA cores"),
        (TF32_TENSOR_TFLOPS, OKABE_ITO["green"], "--", "tf32 tensor cores"),
        (BF16_TENSOR_TFLOPS, TC_COLOR, "-.", "bf16 tensor cores"),
    ]:
        ax.axhline(tflops * 1000.0, color=color, linewidth=1.6, linestyle=style,
                   label=f"{name} — {tflops:g} TFLOP/s")

    # Shade above the absolute roof: the memory diagonal capped by the fastest
    # (bf16) ceiling, so the tensor-core half of the plot is not greyed out.
    absolute_roof = np.minimum(mem_line, BF16_TENSOR_TFLOPS * 1000.0)
    ax.fill_between(xs, absolute_roof, BF16_TENSOR_TFLOPS * 1000.0 * 40,
                    color="#F4F4F4", zorder=0)

    for ridge, tflops, color, label in [
        (RIDGE_FP32, FP32_PEAK_TFLOPS, FP32_COLOR, "fp32"),
        (RIDGE_BF16, BF16_TENSOR_TFLOPS, TC_COLOR, "bf16"),
    ]:
        y = tflops * 1000.0
        ax.plot([ridge], [y], marker="o", markersize=7, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate(
        f"fp32 ridge point\n{RIDGE_FP32:.0f} FLOP/byte\n"
        f"(left = memory-bound, right = compute-bound)",
        (RIDGE_FP32, FP32_PEAK_TFLOPS * 1000.0),
        xytext=(-16, -78), textcoords="offset points", ha="right", fontsize=8.5,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=FP32_COLOR, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=FP32_COLOR, lw=1.0))
    ax.annotate(
        f"bf16 ridge {RIDGE_BF16:.0f} FLOP/byte",
        (RIDGE_BF16, BF16_TENSOR_TFLOPS * 1000.0),
        xytext=(8, 10), textcoords="offset points", fontsize=8.5, color=TC_COLOR)

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    plotted = []
    for i, r in enumerate(pts):
        ai, measured = arithmetic_intensity(r.rec)
        color = TC_COLOR if r.precision != "fp32" else FP32_COLOR
        if r.is_baseline:
            color = BASELINE_COLOR
        ax.plot([ai], [r.gflops], marker=marker_cycle[i % len(marker_cycle)],
                markersize=9, color=color, markeredgecolor="#111111",
                markeredgewidth=0.9, linestyle="none", zorder=6,
                # Hollow marker = compulsory-traffic estimate, not a measurement.
                markerfacecolor=color if measured else "white")
        plotted.append((ai, r.gflops, r.name, color))

    # All kernels at one size share the same compulsory AI, so their labels (not
    # the markers) are pushed apart in log space, with a gap derived from figure geometry.
    y_lo = max(1.0, min(gfs) * 0.25)
    y_hi = BF16_TENSOR_TFLOPS * 1000.0 * 3
    decades = max(math.log10(y_hi / y_lo), 1e-6)
    axes_height_pt = fig.get_figheight() * 72.0 * 0.78   # minus title/xlabel
    pt_per_decade = axes_height_pt / decades
    MIN_LOG_GAP = 11.0 / pt_per_decade                   # ~11pt of clear space
    plotted.sort(key=lambda t: t[1])
    label_y: List[float] = []
    for _ai, y, _name, _c in plotted:
        ly = math.log10(max(y, 1e-9))
        if label_y and ly - label_y[-1] < MIN_LOG_GAP:
            ly = label_y[-1] + MIN_LOG_GAP
        label_y.append(ly)
    for (ai, y, name, color), ly in zip(plotted, label_y):
        y_lab = 10.0 ** ly
        ax.annotate(name, (ai, y), xytext=(ai * 1.30, y_lab),
                    textcoords="data", fontsize=8.5, color="#111111",
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7,
                                    shrinkA=2, shrinkB=1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte of DRAM traffic)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Roofline — DGX Spark (GB10, sm_121) — {size}")
    _mark_synthetic(fig, rows)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left", fontsize=8, ncol=1,
              framealpha=0.92, frameon=True, facecolor="white",
              edgecolor="#DDDDDD")

    any_measured = any(arithmetic_intensity(r.rec)[1] for r in pts)
    fig.text(0.005, -0.02,
             ("Hollow markers: x is the COMPULSORY arithmetic intensity "
              "(each of A, B, C touched exactly once) — an upper bound, not a "
              "measurement, so the true point lies to the LEFT.\n"
              "Log the Nsight metric dram__bytes.sum as \"dram_bytes\" in the "
              "JSONL and these become filled markers at the measured position."
              if not any_measured else
              "Filled markers use measured DRAM traffic (dram_bytes); hollow "
              "markers use the compulsory-traffic upper bound."),
             fontsize=8, color="#555555", va="top")

    path = os.path.join(outdir, "roofline.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_latency(rows: List[Row], size: str, outdir: str) -> Optional[str]:
    """min/p50/p90/p99 glyph per kernel — not a boxplot; the harness logs no quartiles."""
    plt = _style()
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    pts = [r for r in rows if r.status == "ok" and r.rec is not None]
    if not pts:
        print("skipping latency chart: no correct results", file=sys.stderr)
        return None
    pts.sort(key=lambda r: RUNG_INDEX.get(r.name, 10_000))

    fig, ax = plt.subplots(figsize=(max(8.0, 1.25 * len(pts) + 2.5), 5.4))

    for i, r in enumerate(pts):
        rec = r.rec or {}
        mn = float(rec.get("min_ms") or 0.0)
        p50 = float(rec.get("p50_ms") or 0.0)
        p90 = float(rec.get("p90_ms") or 0.0)
        p99 = float(rec.get("p99_ms") or 0.0)
        color = TC_COLOR if r.precision != "fp32" else FP32_COLOR
        if r.is_baseline:
            color = BASELINE_COLOR

        # whisker: min .. p99
        ax.plot([i, i], [mn, p99], color="#333333", linewidth=1.2, zorder=2)
        for y in (mn, p99):
            ax.plot([i - 0.13, i + 0.13], [y, y], color="#333333",
                    linewidth=1.2, zorder=2)
        # body: p50 .. p90
        ax.add_patch(Rectangle((i - 0.28, min(p50, p90)), 0.56,
                               max(abs(p90 - p50), 1e-9),
                               facecolor=color, edgecolor="#222222",
                               linewidth=0.9, alpha=0.85, zorder=3))
        # median tick
        ax.plot([i - 0.30, i + 0.30], [p50, p50], color="#111111",
                linewidth=2.0, zorder=4)

        # Same diagnostic as TimingStats::spread() in stats.hpp.
        if mn > 0:
            ax.annotate(f"p99/min {p99 / mn:.2f}x", (i, p99),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8, color="#444444")

    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([r.name for r in pts], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("iteration latency (ms, CUDA events)")
    # Log scale: kernels can be orders of magnitude apart.
    ax.set_yscale("log")
    ax.set_title(f"Per-iteration latency distribution — {size}")
    _mark_synthetic(fig, pts)
    ax.yaxis.grid(True, which="both", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    legend = [
        Line2D([0], [0], color="#111111", lw=2.0, label="p50 (median)"),
        Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", edgecolor="#222222",
                  label="p50 → p90"),
        Line2D([0], [0], color="#333333", lw=1.2, label="min → p99 (whisker)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5)

    n = (pts[0].rec or {}).get("n")
    fig.text(0.005, -0.04,
             f"Percentiles over {n if n else '?'} timed iterations after warmup, "
             f"each timed individually with cudaEvent (not a loop-and-divide, "
             f"which would destroy the distribution).\nGB10 cannot lock clocks "
             f"(nvidia-smi -lgc is a no-op), so this spread is a property of the "
             f"machine, not a bug in the measurement — which is exactly why it "
             f"gets its own chart.",
             fontsize=8, color="#555555", va="top")

    path = os.path.join(outdir, "latency.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_history(history: List[Tuple[str, List[Dict[str, Any]]]], size: str,
                  outdir: str) -> Optional[str]:
    """GFLOP/s per kernel across successive runs."""
    plt = _style()
    series = [(name, [r for r in recs if status_of(r) == "ok"])
              for name, recs in history]
    series = [(n, r) for n, r in series if len(r) >= 1]
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    palette = [OKABE_ITO[k] for k in
               ("blue", "vermillion", "green", "orange", "purple", "sky", "black")]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (name, recs) in enumerate(series):
        recs = sorted(recs, key=lambda r: (str(r.get("timestamp") or ""), r.get("_line", 0)))
        ys = [float(r.get("gflops_p50") or 0.0) for r in recs]
        # x is the run index, not the timestamp: attempt ordering is what matters.
        ax.plot(range(1, len(ys) + 1), ys, marker=markers[i % len(markers)],
                color=palette[i % len(palette)], linewidth=1.8, markersize=6,
                label=name)
    ax.set_xlabel("run # (chronological, per kernel)")
    ax.set_ylabel("GFLOP/s (p50)")
    ax.set_title(f"Progression across runs — {size}")
    _mark_synthetic(fig, [r for _n, recs in history for r in recs])
    ax.grid(True, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, ncol=2)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(outdir, "history.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ladder table and charts from the harness JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[-1])
    ap.add_argument("jsonl", nargs="?", default=DEFAULT_JSONL,
                    help=f"results file (default: {DEFAULT_JSONL})")
    ap.add_argument("--size", default=None,
                    help="problem size as MxNxK; default is the largest present")
    ap.add_argument("--all-sizes", action="store_true",
                    help="report every problem size in the file, not just one")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table for a README instead of the "
                         "terminal table")
    ap.add_argument("--history", nargs="?", const="__ALL__", default=None,
                    metavar="KERNEL",
                    help="show progression across runs; optionally for one kernel")
    ap.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="where PNGs go")
    args = ap.parse_args(argv)

    rows_all = load_jsonl(args.jsonl)

    # Attention records key on B/H/S/D, not M/N/K — they have their own report.
    attn = [r for r in rows_all if r.get("phase") == "attention"]
    if attn:
        if len(attn) == len(rows_all):
            sys.exit(f"{args.jsonl} holds attention results — use:\n"
                     f"    python3 bench/report_attention.py {args.jsonl}")
        print(f"note: skipping {len(attn)} attention records "
              f"(bench/report_attention.py reads those)", file=sys.stderr)
        rows_all = [r for r in rows_all if r.get("phase") != "attention"]

    synthetic = any_synthetic(rows_all)
    available = sizes_in(rows_all)

    if args.size:
        if args.size not in available:
            sys.exit(f"error: no results for size {args.size}\n"
                     f"       available: {', '.join(available)}")
        sizes = [args.size]
    elif args.all_sizes:
        sizes = available
    else:
        # Small sizes are launch-latency dominated; the largest is the headline.
        sizes = available[:1]
        if len(available) > 1:
            print(f"note: {len(available)} problem sizes in this file "
                  f"({', '.join(available)}); showing {sizes[0]}. "
                  f"Use --size or --all-sizes.", file=sys.stderr)

    os.makedirs(args.outdir, exist_ok=True)
    written: List[str] = []

    for size in sizes:
        latest = latest_per_kernel(rows_all, size)
        rows = build_rows(latest)

        if args.markdown:
            print_markdown(rows, size, synthetic=synthetic)
        else:
            print_table(rows, size, args.jsonl, synthetic=synthetic)

        history = []
        if args.history is not None:
            only = None if args.history == "__ALL__" else args.history
            history = print_history(rows_all, size, only)

        if not args.no_charts:
            try:
                for fn in (chart_ladder, chart_roofline, chart_latency):
                    p = fn(rows, size, args.outdir)
                    if p:
                        written.append(p)
                if history:
                    p = chart_history(history, size, args.outdir)
                    if p:
                        written.append(p)
            except ImportError:
                # Charts are optional; a missing matplotlib must never cost the table.
                print("note: matplotlib not installed, skipping charts "
                      "(pip install matplotlib)", file=sys.stderr)

    if written:
        print("charts written:")
        for p in written:
            print(f"  {p}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
