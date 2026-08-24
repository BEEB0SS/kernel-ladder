"""Shared loading and timing helpers; timing methodology must match the C++ harness (harness.cuh / stats.hpp)."""

from __future__ import annotations

import math
import sys
from typing import Callable, Dict, List, Optional

import torch


def load_extension(verbose: bool = True) -> str:
    """Import the compiled extension (its TORCH_LIBRARY side effect registers torch.ops.kernel_ladder) and return the rung name."""
    try:
        import kernel_ladder_C  # noqa: F401  (imported for its side effects)
    except ImportError as exc:
        sys.exit(
            f"error: could not import the compiled extension ({exc}).\n"
            f"       Build it first:\n"
            f"           cd torch_ext && python setup.py build_ext --inplace\n"
            f"       and run from a directory where the built .so is importable.\n"
            f"       If the build itself failed, the usual causes are a PyTorch\n"
            f"       build without sm_121 support, an ABI mismatch with a stale\n"
            f"       build/ directory, or a missing ninja — see the root README."
        )
    if not hasattr(torch.ops, "kernel_ladder"):
        sys.exit("error: the extension imported but registered no ops. That means "
                 "the TORCH_LIBRARY block in binding.cu did not run — usually a "
                 "stale .so from an older build. Delete build/ and rebuild.")
    name = torch.ops.kernel_ladder.kernel_name()
    if verbose:
        print(f"[kernel-ladder] op bound to rung: {name}")
    return name


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        sys.exit(
            "error: no CUDA device visible to PyTorch.\n"
            "       This op has no CPU implementation by design. On a Spark,\n"
            "       torch.cuda.is_available() == False almost always means a\n"
            "       CPU-only or wrong-arch PyTorch build — see the root README."
        )
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    cc = f"{props.major}.{props.minor}"
    print(f"[kernel-ladder] {props.name}, compute capability {cc}, "
          f"{props.multi_processor_count} SMs")
    if (props.major, props.minor) != (12, 1):
        print(f"[kernel-ladder] WARNING: expected compute capability 12.1 (GB10). "
              f"The tile sizes and the 99KB shared-memory assumptions in this "
              f"repo are tuned for GB10 and may be wrong on a {cc} part.")
    return dev


def percentile(samples: List[float], q: float) -> float:
    """Linearly interpolated percentile — keep in sync with ladder::percentile in stats.hpp."""
    if not samples:
        return 0.0
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def summarize(samples_ms: List[float]) -> Dict[str, float]:
    """min / p50 / p90 / p99 / mean / stddev / cv, matching TimingStats."""
    if not samples_ms:
        return {k: 0.0 for k in
                ("min_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms",
                 "mean_ms", "stddev_ms", "cv", "n")}
    s = sorted(samples_ms)
    n = len(s)
    mean = sum(s) / n
    # Bessel-corrected as in stats.hpp; n == 1 reports 0 (sample variance undefined).
    var = sum((v - mean) ** 2 for v in s) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    return {
        "min_ms": s[0],
        "p50_ms": percentile(s, 0.50),
        "p90_ms": percentile(s, 0.90),
        "p99_ms": percentile(s, 0.99),
        "max_ms": s[-1],
        "mean_ms": mean,
        "stddev_ms": sd,
        "cv": sd / mean if mean > 0 else 0.0,
        "n": float(n),
    }


def time_cuda(fn: Callable[[], object], warmup: int = 25, iters: int = 100,
              flops: Optional[float] = None) -> Dict[str, float]:
    """Time `fn` per iteration with CUDA events; events are preallocated and elapsed_time is read only after the final sync (it forces a sync per event)."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    stats = summarize(samples)
    if flops:
        stats["gflops_p50"] = flops / (stats["p50_ms"] * 1e-3) / 1e9
        stats["gflops_min"] = flops / (stats["min_ms"] * 1e-3) / 1e9
    return stats


def print_stats_header() -> None:
    print(f"\n{'what':<34}{'p50 ms':>9}{'min ms':>9}{'p90 ms':>9}{'p99 ms':>9}"
          f"{'cv':>7}{'GFLOP/s':>10}")
    print("-" * 87)


def print_stats(label: str, st: Dict[str, float]) -> None:
    gf = st.get("gflops_p50")
    print(f"{label:<34}{st['p50_ms']:>9.3f}{st['min_ms']:>9.3f}"
          f"{st['p90_ms']:>9.3f}{st['p99_ms']:>9.3f}{st['cv']:>7.3f}"
          f"{(f'{gf:>10.1f}' if gf else f'{chr(8212):>10}')}"
          + ("   [NOISY: cv>5%]" if st["cv"] > 0.05 else ""))
