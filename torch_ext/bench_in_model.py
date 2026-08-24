#!/usr/bin/env python3
"""Benchmark the custom sgemm op inside a transformer MLP block, with Amdahl's-law analysis."""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_util import (load_extension, require_cuda, print_stats,
                        print_stats_header, time_cuda)


class StockMLPBlock(nn.Module):
    """Baseline: plain nn.Linear transformer feed-forward block."""

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        d_ff = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ff, bias=True)
        self.down = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.up(h)
        h = F.gelu(h)
        h = self.down(h)
        return x + h


class LadderMLPBlock(nn.Module):
    """Same block with both matmuls replaced by the custom op; weights copied from the stock block."""

    def __init__(self, stock: StockMLPBlock) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(stock.norm.normalized_shape)
        self.norm.load_state_dict(stock.norm.state_dict())
        # nn.Linear stores [out, in]; the kernel wants row-major [in, out] -- transpose once here, not per forward.
        self.up_w = nn.Parameter(stock.up.weight.detach().t().contiguous(),
                                 requires_grad=False)
        self.up_b = nn.Parameter(stock.up.bias.detach().clone(), requires_grad=False)
        self.down_w = nn.Parameter(stock.down.weight.detach().t().contiguous(),
                                   requires_grad=False)
        self.down_b = nn.Parameter(stock.down.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = self.norm(x)
        h2 = h.reshape(-1, shape[-1])
        h2 = torch.ops.kernel_ladder.sgemm(h2, self.up_w) + self.up_b
        h2 = F.gelu(h2)
        h2 = torch.ops.kernel_ladder.sgemm(h2, self.down_w) + self.down_b
        return x + h2.reshape(shape)


class Stack(nn.Module):
    """N blocks in sequence; a single block is dominated by launch overhead."""

    def __init__(self, blocks) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Run:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = require_cuda()
    rung = load_extension()

    # Strict fp32 on both sides so this compares kernels, not precision; TF32 is measured separately below.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(0)
    d_model, d_ff = args.d_model, args.d_model * args.expansion
    M = args.batch * args.seq

    stock_blocks = [StockMLPBlock(d_model, args.expansion).to(device).eval()
                    for _ in range(args.layers)]
    ladder_blocks = [LadderMLPBlock(b).to(device).eval() for b in stock_blocks]
    stock = Stack(stock_blocks).to(device).eval()
    ours = Stack(ladder_blocks).to(device).eval()

    x = torch.randn(args.batch, args.seq, d_model, device=device)

    print(f"\n== model ==")
    print(f"   {args.layers} x MLP block, d_model={d_model}, d_ff={d_ff}, "
          f"input [{args.batch}, {args.seq}, {d_model}]")
    print(f"   each block: LayerNorm -> [{M}x{d_model}] @ [{d_model}x{d_ff}] -> "
          f"GELU -> [{M}x{d_ff}] @ [{d_ff}x{d_model}] -> residual")
    print(f"   custom op is rung: {rung}")

    with torch.no_grad():
        y_ref = stock(x)
        y_ours = ours(x)
    denom = y_ref.abs().clamp_min(1e-6)
    err = float(((y_ours - y_ref).abs() / denom).max())
    # Tolerance is looser than test_op.py's 2e-4: error compounds across layers and the residual stream.
    same = torch.allclose(y_ours, y_ref, rtol=2e-3, atol=2e-3)
    print(f"\n== correctness ==")
    print(f"   {'PASS' if same else 'FAIL'}  max rel err vs stock block: {err:.3g}")
    if not same:
        print("   Stop here. Timing a model that computes the wrong thing is "
              "not a benchmark.\n")
        return 1

    print(f"\n== kernel level (the two GEMMs in one block, in isolation) ==")
    a1 = torch.randn(M, d_model, device=device)
    b1 = torch.randn(d_model, d_ff, device=device)
    a2 = torch.randn(M, d_ff, device=device)
    b2 = torch.randn(d_ff, d_model, device=device)

    print_stats_header()
    with torch.no_grad():
        up_ours = time_cuda(lambda: torch.ops.kernel_ladder.sgemm(a1, b1),
                            args.warmup, args.iters, 2.0 * M * d_model * d_ff)
        up_torch = time_cuda(lambda: torch.matmul(a1, b1),
                             args.warmup, args.iters, 2.0 * M * d_model * d_ff)
        dn_ours = time_cuda(lambda: torch.ops.kernel_ladder.sgemm(a2, b2),
                            args.warmup, args.iters, 2.0 * M * d_ff * d_model)
        dn_torch = time_cuda(lambda: torch.matmul(a2, b2),
                             args.warmup, args.iters, 2.0 * M * d_ff * d_model)
    print_stats(f"  up-proj   kernel_ladder", up_ours)
    print_stats(f"  up-proj   torch.matmul", up_torch)
    print_stats(f"  down-proj kernel_ladder", dn_ours)
    print_stats(f"  down-proj torch.matmul", dn_torch)

    gemm_ours = up_ours["p50_ms"] + dn_ours["p50_ms"]
    gemm_torch = up_torch["p50_ms"] + dn_torch["p50_ms"]
    kernel_speedup = gemm_torch / gemm_ours if gemm_ours > 0 else 0.0
    print(f"\n   GEMM time per block: {gemm_torch:.3f} ms stock -> "
          f"{gemm_ours:.3f} ms ours   =  {kernel_speedup:.2f}x KERNEL SPEEDUP")

    print(f"\n== end to end ({args.layers}-block forward pass) ==")
    print_stats_header()
    with torch.no_grad():
        e2e_torch = time_cuda(lambda: stock(x), args.warmup, args.iters)
        e2e_ours = time_cuda(lambda: ours(x), args.warmup, args.iters)
        torch.backends.cuda.matmul.allow_tf32 = True
        e2e_tf32 = time_cuda(lambda: stock(x), args.warmup, args.iters)
        torch.backends.cuda.matmul.allow_tf32 = False
    print_stats("  stock PyTorch (fp32)", e2e_torch)
    print_stats("  kernel_ladder op", e2e_ours)
    print_stats("  stock PyTorch (TF32, default)", e2e_tf32)

    e2e_speedup = e2e_torch["p50_ms"] / e2e_ours["p50_ms"]
    e2e_vs_tf32 = e2e_tf32["p50_ms"] / e2e_ours["p50_ms"]

    # p is measured, not assumed: GEMM fraction of the stock block's total time.
    total_gemm_stock = gemm_torch * args.layers
    p = total_gemm_stock / e2e_torch["p50_ms"] if e2e_torch["p50_ms"] > 0 else 0.0
    p = min(max(p, 0.0), 1.0)
    s = kernel_speedup
    predicted = 1.0 / ((1.0 - p) + p / s) if s > 0 else 1.0
    ceiling = 1.0 / (1.0 - p) if p < 1.0 else float("inf")

    print(f"\n== Amdahl's law ==")
    print(f"   GEMM fraction of the stock forward pass   p = {p:.1%}")
    print(f"   everything else (LayerNorm/GELU/residual/launches) = {1 - p:.1%}")
    print(f"   kernel speedup on that fraction           s = {s:.2f}x")
    print()
    print(f"   predicted end-to-end   1 / ((1-p) + p/s)    = {predicted:.2f}x")
    print(f"   measured  end-to-end                        = {e2e_speedup:.2f}x")
    print(f"   ceiling if the GEMM were FREE  1/(1-p)      = {ceiling:.2f}x")
    print()
    print(f"   vs stock PyTorch defaults (TF32 on)         = {e2e_vs_tf32:.2f}x")

    gap = predicted - e2e_speedup
    print(f"\n   what this means")
    print(f"   ---------------")
    print(f"   A {s:.2f}x kernel bought {e2e_speedup:.2f}x end to end, because "
          f"only {p:.0%} of the")
    print(f"   forward pass was ever in the GEMM. The other {1 - p:.0%} is "
          f"memory-bound elementwise")
    print(f"   work and launch overhead, and it did not get faster because "
          f"nothing about it changed.")
    print(f"   Even an infinitely fast GEMM would stop at {ceiling:.2f}x.")
    if abs(gap) > 0.15 * max(predicted, 1.0):
        print()
        print(f"   NOTE: measured and predicted differ by {gap:+.2f}x, which is "
              f"more than rounding.")
        if gap > 0:
            print(f"   The substitution cost something Amdahl does not model — "
                  f"most likely the extra")
            print(f"   bias-add and reshape in LadderMLPBlock, cache pressure "
                  f"from the transposed")
            print(f"   weights, or lower occupancy leaving less room to overlap "
                  f"the elementwise ops.")
        else:
            print(f"   We beat the prediction, which means something OTHER than "
                  f"GEMM time improved —")
            print(f"   usually better L2 residency for the following elementwise "
                  f"op. Verify with ncu")
            print(f"   before claiming it; a pleasant surprise in a benchmark is "
                  f"usually a measurement bug.")
    print()
    print(f"   Where to go next, in order of payoff on THIS machine:")
    print(f"     1. Fuse. The bias add, GELU and residual are three separate "
          f"passes over the")
    print(f"        activations, each paying full DRAM bandwidth. On GB10 that "
          f"is ~231 GB/s of")
    print(f"        measured bandwidth, so those passes are expensive relative "
          f"to an H100. Fusing")
    print(f"        them attacks the (1-p) that the GEMM work cannot reach.")
    print(f"     2. Raise p by making the model bigger (larger d_model, longer "
          f"sequence). The")
    print(f"        GEMM grows as O(M*d^2) while the elementwise work grows as "
          f"O(M*d), so p rises")
    print(f"        with d. The same kernel is worth more in a bigger model.")
    print(f"     3. Only then, keep optimizing the GEMM.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
