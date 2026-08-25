#!/usr/bin/env python3
"""sweep.py — Read a tile-sweep JSONL and show where the optimum is.

USAGE
    python3 bench/sweep.py                                   # default sweep file
    python3 bench/sweep.py bench/results/sweep_tiles.jsonl
    python3 bench/sweep.py --rows BM,BN --cols TM,TN         # choose the axes
    python3 bench/sweep.py --metric pct_of_peak
    python3 bench/sweep.py --top 10 --no-charts
"""
# The USAGE block above is split out at runtime as the argparse epilog.

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

matplotlib.use("Agg")   # headless PNG output; no display needed

DEFAULT_JSONL = "bench/results/sweep_tiles.jsonl"
DEFAULT_OUTDIR = "bench/results"

# Keep in sync with src/common/gb10.hpp.
FP32_PEAK_GFLOPS = 29.7 * 1000.0
MAX_SMEM_PER_BLOCK_BYTES = 99 * 1024
MAX_THREADS_PER_BLOCK = 1024

TILE_PARAMS = ("BM", "BN", "BK", "TM", "TN")

# Two-char codes stamped into failed heatmap cells; second field is the legend.
FAILURE_CODES = {
    "smem":          ("SM", "over the 99 KB shared-memory cap (GB10)"),
    "threads":       ("TH", "over the 1024 threads/block cap"),
    "geometry":      ("GE", "tile dimensions do not divide evenly"),
    "compile_error": ("CE", "nvcc refused it"),
    "run_error":     ("RE", "built, but the launch or the run failed"),
    "incorrect":     ("XX", "ran, but disagreed with the CPU oracle"),
    "skipped":       ("--", "skipped for another reason"),
    "missing":       ("",   "not in the sweep"),
}

# Harness JSONL comes from fprintf, so bare nan/inf tokens are not legal JSON.
_NONFINITE = re.compile(r':\s*(-?)(?:nan|inf|NAN|INF)(?=\s*[,}])')


def _sanitize(line: str) -> str:
    def repl(m: "re.Match[str]") -> str:
        return ": null" if "nan" in m.group(0).lower() else f": {m.group(1)}Infinity"
    return _NONFINITE.sub(repl, line)


def load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        sys.exit(f"error: no sweep file at {path}\n"
                 f"       run it first:  scripts/sweep_tiles.sh\n"
                 f"       (or: python3 bench/make_synthetic_results.py --sweep "
                 f"for a test fixture)")
    rows: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(_sanitize(raw), parse_constant=lambda _c: None)
        except json.JSONDecodeError:
            print(f"warning: {path}:{lineno} unparseable, skipping", file=sys.stderr)
            continue
        rec["_line"] = lineno
        rows.append(rec)
    if not rows:
        sys.exit(f"error: {path} has no usable records")
    return rows


def classify(rec: Dict[str, Any]) -> str:
    """Bucket a record into 'ok' or a FAILURE_CODES key, re-derived from several fields."""
    status = rec.get("sweep_status")
    reason = str(rec.get("reason") or "").lower()

    if status == "skipped":
        if "shared" in reason or "smem" in reason:
            return "smem"
        if "thread" in reason and "warp" not in reason:
            return "threads"
        if "divis" in reason or "geometry" in reason or "divide" in reason:
            return "geometry"
        return "skipped"
    if status in ("compile_error", "run_error", "incorrect"):
        return status
    if rec.get("skipped"):
        return "skipped"
    if rec.get("correct") is False:
        return "incorrect"
    if not isinstance(rec.get("gflops_p50"), (int, float)) or not rec["gflops_p50"] > 0:
        return "run_error"
    return "ok"


def latest_per_config(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One record per tile config: the file is append-only, so the newest wins."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cfg = r.get("config") or "_".join(
            f"{p}{r.get(p)}" for p in TILE_PARAMS if r.get(p) is not None)
        if not cfg:
            continue
        key = (str(r.get("timestamp") or ""), r.get("_line", 0))
        prev = best.get(cfg)
        if prev is None or key > (str(prev.get("timestamp") or ""), prev.get("_line", 0)):
            r = dict(r)
            r["config"] = cfg
            best[cfg] = r
    return best


def varying_params(recs: Sequence[Dict[str, Any]]) -> List[str]:
    """Tile parameters that take more than one value in these records."""
    out = []
    for p in TILE_PARAMS:
        vals = {r.get(p) for r in recs if r.get(p) is not None}
        if len(vals) > 1:
            out.append(p)
    return out


def choose_axes(recs: Sequence[Dict[str, Any]],
                rows_arg: Optional[str],
                cols_arg: Optional[str]) -> Tuple[List[str], List[str]]:
    """Pick axis params: block tile (BM,BN,BK) on rows, thread tile (TM,TN) on cols."""
    varying = varying_params(recs)
    if rows_arg or cols_arg:
        rows = [p.strip().upper() for p in (rows_arg or "").split(",") if p.strip()]
        cols = [p.strip().upper() for p in (cols_arg or "").split(",") if p.strip()]
        bad = [p for p in rows + cols if p not in TILE_PARAMS]
        if bad:
            sys.exit(f"error: unknown tile parameter(s) {bad}; "
                     f"pick from {', '.join(TILE_PARAMS)}")
        unplaced = [p for p in varying if p not in rows + cols]
        if unplaced:
            sys.exit(f"error: {', '.join(unplaced)} vary in this sweep but are on "
                     f"neither axis, so several configs would land in the same "
                     f"cell.\n       Add them to --rows/--cols, or filter the "
                     f"file first.")
        return rows or ["BM"], cols or ["TM"]

    rows = [p for p in ("BM", "BN", "BK") if p in varying]
    cols = [p for p in ("TM", "TN") if p in varying]
    # Degenerate sweeps: keep the grid 2D by giving each axis at least one param.
    if not rows:
        rows = [p for p in ("BM", "BN", "BK") if any(r.get(p) is not None for r in recs)][:1]
    if not cols:
        cols = [p for p in ("TM", "TN") if any(r.get(p) is not None for r in recs)][:1]
    return rows or ["BM"], cols or ["TM"]


def axis_key(rec: Dict[str, Any], params: Sequence[str]) -> Tuple:
    return tuple(rec.get(p) for p in params)


def axis_label(key: Tuple, params: Sequence[str]) -> str:
    return " ".join(f"{p}{v}" for p, v in zip(params, key))


def print_summary(cfgs: Dict[str, Dict[str, Any]], metric: str, top: int,
                  path: str, synthetic: bool) -> None:
    if synthetic:
        banner = ("!!! SYNTHETIC SWEEP DATA — invented by "
                  "bench/make_synthetic_results.py --sweep. Not measurements. !!!")
        print("\n" + "*" * len(banner) + f"\n{banner}\n" + "*" * len(banner))

    ok = [r for r in cfgs.values() if classify(r) == "ok"]
    failed = [r for r in cfgs.values() if classify(r) != "ok"]

    print(f"\n== tile sweep for 04_2d_blocktile ==  (source: {path})")
    sizes = sorted({f"{r.get('M')}x{r.get('N')}x{r.get('K')}" for r in cfgs.values()})
    print(f"   {len(cfgs)} configs, {len(ok)} ran, {len(failed)} did not"
          f"   problem size(s): {', '.join(sizes)}")

    if not ok:
        print("\n   NO CONFIG PRODUCED A NUMBER. The failure breakdown below is "
              "the whole result;\n   the heatmap will show only the reasons.")
    else:
        ok.sort(key=lambda r: -float(r.get(metric) or 0.0))
        print(f"\n   top {min(top, len(ok))} by {metric}:")
        print(f"   {'#':>2} {'config':<30}{'GFLOP/s':>10}{'% peak':>8}"
              f"{'p50 ms':>9}{'cv':>7}{'regs':>6}{'spill':>7}{'smem':>7}{'thr':>6}")
        best = float(ok[0].get(metric) or 0.0)
        for i, r in enumerate(ok[:top], 1):
            val = float(r.get(metric) or 0.0)
            gf = float(r.get("gflops_p50") or 0.0)
            spill = int(r.get("spill_bytes") or 0)
            print(f"   {i:>2} {r['config']:<30}{gf:>10.1f}"
                  f"{100.0 * gf / FP32_PEAK_GFLOPS:>7.1f}%"
                  f"{float(r.get('p50_ms') or 0):>9.3f}"
                  f"{float(r.get('cv') or 0):>7.3f}"
                  f"{int(r.get('registers_per_thread') or 0):>6}"
                  f"{spill:>7}"
                  f"{int(r.get('smem_bytes') or 0) // 1024:>6}K"
                  f"{int(r.get('threads') or 0):>6}"
                  + ("   <-- best" if i == 1 else
                     ("   (within noise of best)" if best > 0 and
                      val / best > 1.0 - max(float(r.get("cv") or 0.0), 0.02)
                      else "")))

        if len(ok) > 1:
            top_gf = float(ok[0].get("gflops_p50") or 0.0)
            second = float(ok[1].get("gflops_p50") or 0.0)
            margin = (top_gf / second - 1.0) if second > 0 else 0.0
            noise = max(float(ok[0].get("cv") or 0.0), float(ok[1].get("cv") or 0.0))
            verdict = ("CLEAR" if margin > 2 * noise else
                       "NOT SIGNIFICANT — rerun both before picking")
            print(f"\n   best beats runner-up by {margin:+.1%}; "
                  f"run-to-run cv is {noise:.1%}  ->  {verdict}")

        spillers = [r for r in ok if int(r.get("spill_bytes") or 0) > 0]
        if spillers:
            print(f"\n   {len(spillers)} config(s) SPILLED registers to local "
                  f"memory. A spill turns a register read into a cache/DRAM "
                  f"read;\n   those configs are slow for a reason that has "
                  f"nothing to do with the tiling logic. Lower TM*TN.")

    if failed:
        print("\n   why configs did not run:")
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in failed:
            buckets.setdefault(classify(r), []).append(r)
        for kind, recs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            code, desc = FAILURE_CODES.get(kind, ("??", kind))
            print(f"   [{code}] {desc}: {len(recs)}")
            for r in recs[:4]:
                reason = str(r.get("reason") or "").strip()
                print(f"        {r['config']:<30} {reason[:78]}")
            if len(recs) > 4:
                print(f"        ... and {len(recs) - 4} more")
    print()


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "legend.frameon": False,
    })
    return plt


def chart_heatmap(cfgs: Dict[str, Dict[str, Any]], row_params: List[str],
                  col_params: List[str], metric: str, outdir: str,
                  synthetic: bool) -> Optional[str]:
    plt = _style()

    recs = list(cfgs.values())
    row_keys = sorted({axis_key(r, row_params) for r in recs},
                      key=lambda k: tuple(-1 if v is None else v for v in k))
    col_keys = sorted({axis_key(r, col_params) for r in recs},
                      key=lambda k: tuple(-1 if v is None else v for v in k))
    ri = {k: i for i, k in enumerate(row_keys)}
    ci = {k: i for i, k in enumerate(col_keys)}

    grid = np.full((len(row_keys), len(col_keys)), np.nan)
    kinds: List[List[str]] = [["missing"] * len(col_keys) for _ in row_keys]
    lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for r in recs:
        i, j = ri[axis_key(r, row_params)], ci[axis_key(r, col_params)]
        kind = classify(r)
        kinds[i][j] = kind
        lookup[(i, j)] = r
        if kind == "ok":
            v = r.get(metric)
            if isinstance(v, (int, float)):
                grid[i, j] = float(v)

    fig_w = max(7.0, 1.35 * len(col_keys) + 4.5)
    fig_h = max(3.8, 0.55 * len(row_keys) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    finite = grid[np.isfinite(grid)]
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="viridis", aspect="auto",
                   origin="upper",
                   vmin=float(finite.min()) if finite.size else 0.0,
                   vmax=float(finite.max()) if finite.size else 1.0)
    im.cmap.set_bad("#E8E8E8")

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels([axis_label(k, col_params) for k in col_keys],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([axis_label(k, row_params) for k in row_keys], fontsize=9)
    ax.set_xlabel("thread tile  (registers per thread)" if col_params == ["TM", "TN"]
                  else " x ".join(col_params))
    ax.set_ylabel("block tile  (shared memory per block)"
                  if row_params[:2] == ["BM", "BN"] else " x ".join(row_params))

    best_ij = None
    if finite.size:
        best_ij = np.unravel_index(np.nanargmax(grid), grid.shape)

    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            kind = kinds[i][j]
            if kind == "ok":
                v = grid[i, j]
                rel = ((v - float(finite.min())) /
                       (float(finite.max()) - float(finite.min()) + 1e-12))
                color = "#111111" if rel > 0.55 else "#FFFFFF"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=8.5, color=color,
                        fontweight="bold" if best_ij == (i, j) else "normal")
                rec = lookup.get((i, j), {})
                if int(rec.get("spill_bytes") or 0) > 0:
                    ax.text(j, i + 0.32, "spill", ha="center", va="center",
                            fontsize=7, color=color, style="italic")
            else:
                code = FAILURE_CODES.get(kind, ("??", ""))[0]
                if code:
                    ax.text(j, i, code, ha="center", va="center", fontsize=9,
                            color="#8A8A8A", fontweight="bold")

    if best_ij is not None:
        ax.add_patch(Rectangle((best_ij[1] - 0.5, best_ij[0] - 0.5), 1, 1,
                               fill=False, edgecolor="#D55E00", linewidth=3.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(metric.replace("_", " "))

    ax.set_title("Tile sweep — 04_2d_blocktile — DGX Spark (GB10, sm_121)")

    present = {k for row in kinds for k in row if k not in ("ok", "missing")}
    handles = [Patch(facecolor="#E8E8E8", edgecolor="#BBBBBB",
                     label=f"[{FAILURE_CODES[k][0]}] {FAILURE_CODES[k][1]}")
               for k in FAILURE_CODES if k in present]
    if handles:
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(0.0, -0.16), fontsize=8, ncol=2)

    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC\nTEST DATA", fontsize=48, color="#D55E00",
                 alpha=0.16, ha="center", va="center", rotation=24,
                 fontweight="bold", zorder=100)

    fig.text(0.005, -0.03 if not handles else -0.24,
             "Orange box = best measured config. Grey cells could not run and "
             "carry a two-letter reason code.\nRead the SHAPE: a broad warm "
             "plateau means the choice does not matter much; one hot cell among "
             "cold ones is usually noise — rerun it before trusting it.",
             fontsize=8, color="#555555", va="top")

    path = os.path.join(outdir, "sweep_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_marginals(cfgs: Dict[str, Dict[str, Any]], metric: str,
                    outdir: str, synthetic: bool) -> Optional[str]:
    """Per-parameter marginal panels: one dot per config, median trend per value."""
    plt = _style()
    ok = [r for r in cfgs.values() if classify(r) == "ok"]
    if not ok:
        return None
    varying = varying_params(ok) or list(TILE_PARAMS)

    fig, axes = plt.subplots(1, len(varying),
                             figsize=(2.55 * len(varying) + 1.2, 3.6),
                             sharey=True)
    if len(varying) == 1:
        axes = [axes]
    for ax, p in zip(axes, varying):
        xs = sorted({r[p] for r in ok if r.get(p) is not None})
        pos = {v: i for i, v in enumerate(xs)}
        for r in ok:
            v = r.get(p)
            if v is None:
                continue
            ax.plot(pos[v], float(r.get(metric) or 0.0), "o", markersize=5,
                    color="#0072B2", alpha=0.55,
                    markeredgecolor="white", markeredgewidth=0.5)
        meds = []
        for v in xs:
            vals = sorted(float(r.get(metric) or 0.0) for r in ok if r.get(p) == v)
            meds.append(vals[len(vals) // 2] if vals else 0.0)
        ax.plot(range(len(xs)), meds, "-", color="#D55E00", linewidth=2.0,
                marker="s", markersize=5, label="median")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel(p)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(fontsize=8)
    fig.suptitle("Tile parameters, one at a time (orange = median)",
                 fontsize=12, fontweight="bold")
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA", fontsize=30, color="#D55E00",
                 alpha=0.16, ha="center", va="center", rotation=18,
                 fontweight="bold", zorder=100)
    fig.text(0.005, -0.06,
             "Each dot is one configuration. A flat median means this knob does "
             "not matter on GB10; vertical spread at one x means the OTHER knobs "
             "dominate there.",
             fontsize=8, color="#555555", va="top")
    path = os.path.join(outdir, "sweep_marginals.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Heatmap and summary for a tile-size sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[-1])
    ap.add_argument("jsonl", nargs="?", default=DEFAULT_JSONL)
    ap.add_argument("--rows", default=None,
                    help="tile params on the y axis, comma separated "
                         "(default: BM,BN,BK — whichever vary)")
    ap.add_argument("--cols", default=None,
                    help="tile params on the x axis (default: TM,TN)")
    ap.add_argument("--metric", default="gflops_p50",
                    choices=["gflops_p50", "gflops_min", "pct_of_peak", "p50_ms"],
                    help="what to color cells by (default: gflops_p50)")
    ap.add_argument("--top", type=int, default=8, help="how many winners to list")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)

    rows = load(args.jsonl)
    cfgs = latest_per_config(rows)
    synthetic = any(bool(r.get("synthetic")) for r in cfgs.values())

    if args.metric == "p50_ms":
        print("note: coloring by p50_ms means LOWER is better, so the heatmap's "
              "bright end is the SLOW end. gflops_p50 is usually the clearer "
              "choice.", file=sys.stderr)

    print_summary(cfgs, args.metric, args.top, args.jsonl, synthetic)

    row_params, col_params = choose_axes(list(cfgs.values()), args.rows, args.cols)

    if not args.no_charts:
        os.makedirs(args.outdir, exist_ok=True)
        written = [chart_heatmap(cfgs, row_params, col_params, args.metric,
                                 args.outdir, synthetic),
                   chart_marginals(cfgs, args.metric, args.outdir, synthetic)]
        written = [w for w in written if w]
        if written:
            print("charts written:")
            for w in written:
                print(f"  {w}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
