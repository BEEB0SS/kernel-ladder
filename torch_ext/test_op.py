#!/usr/bin/env python3
"""Correctness tests and benchmark for the kernel_ladder.sgemm PyTorch op."""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

from bench_util import (load_extension, require_cuda, print_stats,
                        print_stats_header, time_cuda)


# 2e-4 matches registry.cu (keep in sync); our kernel and cuBLAS sum K in different orders.
RTOL = 2e-4
ATOL = 2e-4


def rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """Max relative error, with the same denominator guard as compare() in C++."""
    denom = want.abs().clamp_min(1e-6)
    return float(((got - want).abs() / denom).max())


def test_correctness(device: torch.device) -> bool:
    print("\n== correctness vs torch.matmul ==")
    ok_all = True

    cases = [
        # (M, N, K, alpha, beta, note)
        (128, 128, 128, 1.0, 0.0, "small square"),
        (512, 512, 512, 1.0, 0.0, "square"),
        (1024, 2048, 512, 1.0, 0.0, "NON-SQUARE — catches M/N mixups"),
        (2048, 512, 1024, 1.0, 0.0, "non-square, the other way"),
        (257, 129, 65, 1.0, 0.0, "primes — exercises every bounds guard"),
        (1000, 1000, 1000, 1.0, 0.0, "not a multiple of any tile size"),
        (512, 512, 512, 2.5, 0.0, "alpha != 1"),
        (512, 512, 512, 1.0, 0.0, "alpha=1 beta=0 baseline"),
        (512, 512, 512, 0.75, 1.5, "alpha and beta both active, with C"),
        (1, 4096, 4096, 1.0, 0.0, "M=1 — a GEMV in GEMM clothing"),
        (4096, 1, 4096, 1.0, 0.0, "N=1"),
    ]

    for (M, N, K, alpha, beta, note) in cases:
        torch.manual_seed(0)
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        c = torch.randn(M, N, device=device, dtype=torch.float32) if beta != 0.0 else None

        got = torch.ops.kernel_ladder.sgemm(a, b, alpha, beta, c)
        want = alpha * (a @ b)
        if c is not None:
            want = want + beta * c

        err = rel_error(got, want)
        passed = torch.allclose(got, want, rtol=RTOL, atol=ATOL)
        ok_all &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {M:>5}x{N:<5}x{K:<5} "
              f"alpha={alpha:<5} beta={beta:<5} max rel err {err:.3g}   {note}")
        if not passed:
            bad = (got - want).abs().argmax()
            print(f"        worst element {int(bad)}: got {got.flatten()[bad]:.6g}, "
                  f"want {want.flatten()[bad]:.6g}")

    print("\n== input validation (each of these must raise) ==")
    a = torch.randn(64, 32, device=device)
    b = torch.randn(32, 64, device=device)
    bad_cases = [
        ("dtype mismatch (fp16)", lambda: torch.ops.kernel_ladder.sgemm(a.half(), b.half())),
        ("inner-dimension mismatch",
         lambda: torch.ops.kernel_ladder.sgemm(a, torch.randn(16, 64, device=device))),
        ("3-D input", lambda: torch.ops.kernel_ladder.sgemm(a.unsqueeze(0), b)),
        ("CPU tensor", lambda: torch.ops.kernel_ladder.sgemm(a.cpu(), b.cpu())),
        ("beta without C", lambda: torch.ops.kernel_ladder.sgemm(a, b, 1.0, 1.0)),
        ("requires_grad", lambda: torch.ops.kernel_ladder.sgemm(
            a.clone().requires_grad_(True), b)),
    ]
    for name, fn in bad_cases:
        try:
            fn()
            print(f"  FAIL  {name}: did NOT raise — the op accepted bad input")
            ok_all = False
        except Exception as exc:                 # noqa: BLE001 — any raise is a pass
            first = str(exc).strip().splitlines()[0][:96]
            print(f"  PASS  {name}: raised — {first}")

    # A transposed tensor is a strided view; exercises the op's .contiguous() handling.
    print("\n== non-contiguous input ==")
    at = torch.randn(512, 256, device=device).t()      # 256x512, strided
    bb = torch.randn(512, 128, device=device)
    assert not at.is_contiguous()
    got = torch.ops.kernel_ladder.sgemm(at, bb)
    want = at @ bb
    passed = torch.allclose(got, want, rtol=RTOL, atol=ATOL)
    ok_all &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  transposed A view, "
          f"max rel err {rel_error(got, want):.3g}")
    return ok_all


class TinyBlock(nn.Module):
    """Calls the custom op; weight is stored pre-transposed (d_in x d_out) to avoid a per-forward transpose."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight_t = nn.Parameter(
            torch.randn(d_in, d_out) / (d_in ** 0.5), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(d_out), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fold leading dims into M; the kernel is 2-D.
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = torch.ops.kernel_ladder.sgemm(x2, self.weight_t)
        return (y + self.bias).reshape(*shape[:-1], self.weight_t.shape[1])


def test_module(device: torch.device) -> bool:
    print("\n== composition inside an nn.Module ==")
    torch.manual_seed(1)
    d_in, d_out = 512, 1024
    block = TinyBlock(d_in, d_out).to(device).eval()

    x = torch.randn(8, 128, d_in, device=device)     # [batch, tokens, features]
    with torch.no_grad():
        got = block(x)
        want = (x.reshape(-1, d_in) @ block.weight_t + block.bias).reshape(8, 128, d_out)

    passed = got.shape == want.shape and torch.allclose(got, want, rtol=RTOL, atol=ATOL)
    print(f"  {'PASS' if passed else 'FAIL'}  3-D input {tuple(x.shape)} -> "
          f"{tuple(got.shape)}, max rel err {rel_error(got, want):.3g}")

    # fullgraph=True: a graph break here means the Meta kernel in binding.cu is missing or wrong.
    compiled_ok = True
    try:
        compiled = torch.compile(block, fullgraph=True)
        with torch.no_grad():
            got_c = compiled(x)
        compiled_ok = torch.allclose(got_c, got, rtol=RTOL, atol=ATOL)
        print(f"  {'PASS' if compiled_ok else 'FAIL'}  torch.compile(fullgraph=True) "
              f"traced the op without a graph break")
    except Exception as exc:                          # noqa: BLE001
        print(f"  FAIL  torch.compile raised: {str(exc).splitlines()[0][:140]}")
        print("        A graph break here almost always means the Meta kernel is "
              "missing or wrong.\n        Check TORCH_LIBRARY_IMPL(kernel_ladder, "
              "Meta, ...) in binding.cu.")
        compiled_ok = False

    return passed and compiled_ok


def benchmark(device: torch.device, shapes, warmup: int, iters: int) -> None:
    print("\n== benchmark: custom op vs torch.matmul (cuBLAS) ==")
    print(f"   warmup={warmup} iters={iters}, per-iteration cudaEvent timing, "
          f"percentiles as in stats.hpp")

    # TF32 off for an apples-to-apples fp32 comparison; the TF32 number is reported separately.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print_stats_header()
    for (M, N, K) in shapes:
        torch.manual_seed(0)
        a = torch.randn(M, K, device=device, dtype=torch.float32)
        b = torch.randn(K, N, device=device, dtype=torch.float32)
        flops = 2.0 * M * N * K
        tag = f"{M}x{N}x{K}"

        st_ours = time_cuda(lambda: torch.ops.kernel_ladder.sgemm(a, b),
                            warmup, iters, flops)
        st_torch = time_cuda(lambda: torch.matmul(a, b), warmup, iters, flops)

        torch.backends.cuda.matmul.allow_tf32 = True
        st_tf32 = time_cuda(lambda: torch.matmul(a, b), warmup, iters, flops)
        torch.backends.cuda.matmul.allow_tf32 = False

        print_stats(f"  {tag}  kernel_ladder.sgemm", st_ours)
        print_stats(f"  {tag}  torch.matmul (fp32)", st_torch)
        print_stats(f"  {tag}  torch.matmul (TF32)", st_tf32)

        speedup = st_torch["p50_ms"] / st_ours["p50_ms"]
        vs_tf32 = st_tf32["p50_ms"] / st_ours["p50_ms"]
        noise = max(st_ours["cv"], st_torch["cv"])
        verdict = ("" if abs(speedup - 1.0) > 2 * noise else
                   "   (within run-to-run noise — not a real difference)")
        print(f"  {tag}  -> {speedup:.2f}x vs fp32 matmul, "
              f"{vs_tf32:.2f}x vs TF32 matmul{verdict}\n")

    print("  Reading these: the fp32 row is the apples-to-apples comparison "
          "(same arithmetic).\n  The TF32 row is what a stock PyTorch model "
          "gets by default — the practical\n  bar — but it is a different, "
          "lower-precision computation, so beating it on\n  speed alone is "
          "not the same as being better.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Run:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="1024,2048,4096",
                    help="comma-separated square sizes, or MxNxK triples")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--skip-bench", action="store_true",
                    help="correctness only (fast)")
    args = ap.parse_args()

    shapes = []
    for tok in args.shapes.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if "x" in tok:
            m, n, k = (int(v) for v in tok.split("x"))
            shapes.append((m, n, k))
        else:
            n = int(tok)
            shapes.append((n, n, n))

    device = require_cuda()
    load_extension()

    ok = test_correctness(device)
    ok &= test_module(device)

    if not ok:
        print("\n*** CORRECTNESS FAILED. Not running the benchmark. ***")
        print("    A fast wrong kernel scores zero — same rule as the C++ "
              "harness.\n")
        return 1

    if not args.skip_bench:
        benchmark(device, shapes, args.warmup, args.iters)

    print("all correctness tests passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
