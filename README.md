# kernel-ladder

CUDA kernels for the two layers that dominate transformer inference, SGEMM and
attention, taken from a naive implementation to faster than fp32 cuBLAS on an
NVIDIA DGX Spark (GB10 Grace Blackwell, sm_121).

I built this as a ladder. Each kernel changes exactly one thing from the one
before it, every result is checked against a CPU oracle before it's timed, and
a step isn't done until Nsight Compute can name the metric that changed. The
point isn't just the fastest kernel at the top. It's the record of what each
step bought, and why.

**Headline numbers** (p50 over 100 iterations, exclusive GPU):

- **SGEMM 4096³:** naive 222 GFLOP/s → **20,549 GFLOP/s** with bf16 tensor
  cores. That's 1.20× fp32 cuBLAS on the same problem, and 38.2 TFLOP/s at 2048³.
- **Attention 4×8×4096×64:** naive 275 GFLOP/s → **35,153 GFLOP/s**
  (FlashAttention with both matmuls on tensor cores). 127×.
- At S=32768 the naive attention can't even allocate its 137 GB score matrix.
  The flash kernel runs it in 4.9 s per iteration.

---

## How I approached it

The order mattered as much as the kernels did:

1. **CPU reference first.** Before writing any CUDA I wrote `sgemm_cpu` and
   `attention_cpu`: deliberately simple, accumulating in double so they
   out-precision everything they judge. They live under host-only unit tests
   (`make test-host`), and every GPU number in this repo is checked against
   them before it's timed.
2. **Then the harness.** 25 warmup iterations, 100 timed ones bracketed by
   CUDA events, p50/p90/p99 and coefficient of variation, SM clock sampled
   before and after, every run appended to a JSONL log. Nothing here was ever
   timed single-shot.
3. **Then the naive kernel, for correctness.** It was the first GPU code
   checked against the oracle, and it's the floor every later kernel is
   measured against. Speed wasn't the point yet.
4. **Then cuBLAS, as the number to beat.** Strict fp32 and TF32, both run in
   every session, profiled with Nsight Systems so I knew exactly which kernels
   I was up against before claiming anything.
5. **Then one change per rung.** Profile with Nsight Compute, name the
   bottleneck, change one thing, re-verify, re-measure, log what moved.
   Coalescing, then shared-memory tiling, then register tiling, then
   vectorization, then tensor cores. Each step's profile is what motivated the
   next one.
6. **Finally, a PyTorch custom op**, to see whether a microbenchmark speedup
   survives inside a real model. Against the fp32 baseline it does (1.10×
   end-to-end in a 4-layer MLP). Against PyTorch's TF32 defaults it doesn't,
   and the benchmark prints both.

## Results: SGEMM

4096×4096×4096, p50 GFLOP/s, every kernel verified against the oracle first.

| # | Kernel | The one change | GFLOP/s | vs prev | vs cuBLAS fp32 |
|---|--------|----------------|--------:|--------:|---------------:|
| 0 | `00_naive` | one thread per output; threads index the strided dimension | 222 | — | 0.01x |
| 1 | `01_coalesced` | `threadIdx.x` indexes the contiguous (column) dimension | 1,016 | 4.58x | 0.06x |
| 2 | `02_smem_tiled` | 32×32×32 tiles of A and B staged in shared memory | 1,669 | 1.64x | 0.10x |
| 3 | `03_1d_blocktile` | TM=8 outputs per thread; one hoisted B read feeds 8 FMAs | 5,299 | 3.18x | 0.31x |
| 4 | `04_2d_blocktile` | 8×8 register tile per thread + a 32-config tile sweep | 10,049 | 1.90x | 0.59x |
| 5 | `05_vectorized` | `float4` for every load/store + transposed A tile | 13,198 | 1.31x | 0.77x |
| 6 | `06_tensorcore` | bf16 `mma.sync` + `ldmatrix` + 2-stage `cp.async` | 20,549 | 1.56x | **1.20x** |

Baselines on the same harness:

- **cuBLAS fp32 (strict): 17,142 GFLOP/s**, 57.7% of the 29.7 TFLOP/s fp32 ceiling
- **cuBLAS TF32: 40,304 GFLOP/s**

The bf16 rung gets compared against both. It beats fp32 cuBLAS at the same
output, and it reaches 51% of cuBLAS TF32, which is its real precision-class
baseline.

![SGEMM ladder](bench/results/ladder.png)

![Roofline](bench/results/roofline.png)

**About rung 4.** Its number came out of a 32-config tile sweep, and the
entire win over the textbook default was one parameter: BK 8 → 16. Half the
barriers per FLOP, twice the work per pipeline stage, 5,178 → 10,049 GFLOP/s.

Every BK=8 config finished at or below 8.6 TFLOP/s, and every TM=TN=4 config
landed in the bottom third. On this part, arithmetic density beats occupancy.

![Tile sweep heatmap](bench/results/sweep_heatmap.png)

## Results: attention

B=4, H=8, D=64, non-causal, p50 at S=4096. The full S-sweep and the causal
numbers are in `bench/results/attention.jsonl`.

| # | Kernel | The one change | GFLOP/s | vs prev |
|---|--------|----------------|--------:|--------:|
| 0 | `00_naive_attention` | 3 kernels; materializes the B×H×S×S score matrix | 275 | — |
| 1 | `01_fused_softmax` | QKᵀ + softmax in one kernel, warp-shuffle reductions | 213 | **0.77x — a measured loss** |
| 2 | `02_flash_tiled` | online softmax, K/V tiled, O(S) memory | 2,058 | 9.7x |
| 3 | `03_flash_tensorcore` | both matmuls on bf16 `mma.sync` | 35,153 | 17.1x |

Two things worth calling out:

- **Rung 1 is a measured loss, and I kept it.** The profiler shows its writes
  shrink 17× while its K-read traffic grows 5×: one block per query row
  re-streams all of K, with nothing amortizing it. Fusion without locality
  doesn't win, and that's exactly what motivates the tiling in rung 2.
- **Causal is a real 2×** (75.9 → 39.0 ms at S=4096), because fully-masked
  K/V tiles are skipped rather than computed and thrown away.

![Attention ladder](bench/results/attention_ladder.png)

![Attention throughput vs S](bench/results/attention_scurve.png)

## What the profiler named at each step

A rung wasn't finished when it got faster. It was finished when Nsight Compute
could name what changed. Measured at 2048³ (SGEMM) and 2×4×1024×64 (attention)
unless noted, on an exclusive GPU.

| SGEMM step | Metric | Before → after | What it says |
|---|---|---|---|
| coalesced | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | 16.5 → 2.5 | B reads went from a 32-sector scatter to 4 contiguous sectors; total load sectors 8.86e9 → 1.34e9 |
| smem tiled | `lts__t_sectors_op_read_lookup_miss.sum` (≈DRAM reads) | 5.84M → 3.11M | only 1.9×, not the textbook 10×: a 16 MB matrix fits GB10's 24 MB L2, which was already absorbing the re-reads. The real win was LSU relief: SoL 70% with the FMA pipe at 8% |
| 1D blocktile | `smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | 8.0% → 20.4% | more arithmetic per issued load; 49 registers, zero spills |
| 2D blocktile | `launch__registers_per_thread` / `sm__warps_active` | 128 regs → 31% occupancy | latency hiding collapsed, so the rung tied its predecessor. The sweep's BK 8→16 restored it: 5,178 → 10,049 GFLOP/s |
| vectorized | `sm__throughput.avg.pct_of_peak_sustained_elapsed` + SASS | 21.3% → 37.0% | pure instruction count: 18× `LDG.E.128`, 16× `STG.E.128`, zero scalar global accesses. Bank conflicts were *unchanged* (33.6M, on the B-tile reads), so the transpose's real win is the `float4` A-tile reads, not the conflicts the folklore predicts |
| tensor cores | `sm__inst_executed_pipe_tensor.sum` | 4,194,304 | exactly 2048³/(16·8·16): every MAC went through the tensor pipe. Route A (WMMA) → Route B (raw `mma.sync`+`ldmatrix`) → +`cp.async`: 16.1 → 19.5 → 20.5 TFLOP/s at 4096³ |

| Attention step | Metric | Before → after | What it says |
|---|---|---|---|
| fused softmax | `lts__t_sectors_op_write.sum` / `lts__t_sectors_op_read.sum` | writes 19.2M → 1.12M; reads 57.9M → 289.7M | the fusion delivered the 17× write cut it promised and lost anyway: 5× the read traffic, because one block per query row re-streams all of K. Fusion without locality is the case for tiling |
| flash (online softmax) | wall time, causal vs full at S=4096 | 75.9 → 39.0 ms | fully-masked K/V tiles are skipped, not computed and discarded; the S×S matrix no longer exists (79 registers, zero spills) |
| flash + tensor cores | `sm__inst_executed_pipe_tensor.sum` | 2,097,152 | exactly FLOPs/(2·16·8·16). The register-permutation path is bit-identical to the shared-memory round trip it replaced; the `cp.async` pipeline added +64% |

GB10 exposes no `dram__*` or hmma-cycle counters, which is why the L2 (`lts__`)
and tensor-instruction counters stand in for them. More in the hardware notes.

## The memory wall, demonstrated

This is why FlashAttention exists, reproduced on this machine's 121 GB of
unified memory:

```
$ ./build/attention --size 4x8x32768x64 --only 00_naive_attention --no-verify
Cannot allocate the 137.44 GB B*H*S*S score matrix (87.57 GB free).
    at S=65536 it would be 549.756 GB; it grows as S^2

$ ./build/attention --size 4x8x32768x64 --only 02_flash_tiled --no-verify
02_flash_tiled    fp32    4931.699 ms    1783.6 GFLOP/s    ok
```

S=8192 (an 8.6 GB score matrix) still *succeeds* on this box, so I pushed the
demo to the point where the O(S²) allocation actually fails while the O(S)
kernel keeps going.

## Keeping the numbers honest

- **A wrong kernel is never timed.** The double-accumulated CPU oracle checks
  every kernel before the benchmark runs. Tolerances are per-precision (2e-4
  fp32, 2e-2 bf16), and the measured max relative error is published beside
  every GFLOP/s.
- **The error metric had to be designed, not assumed.** Pure per-element
  relative error is unsatisfiable for GEMM with mixed-sign data. Even cuBLAS
  "fails" it, because outputs near zero carry accumulation noise that's huge
  relative to that one element while being ~1e-7 of the problem scale. So I
  floor the denominator at the problem's natural output scale, √K.
- **No single-shot timing.** 25 warmup + 100 timed iterations with CUDA
  events; p50/p90/p99 and cv reported; cv > 5% gets flagged instead of
  averaged away.
- **Clocks are logged, not locked.** GB10 can't lock clocks (`nvidia-smi -lgc`
  is a no-op), so every result records the SM clock before and after. A
  speedup that arrived with a clock increase isn't a speedup.
- **The GPU has to be exclusive.** I learned this the hard way: a background
  inference server halved throughput in a way that looked exactly like thermal
  throttling. The protocol is now `docker ps` + `nvidia-smi` for compute
  processes before any session.
- **Baselines match precision class.** Strict-fp32 cuBLAS and TF32 cuBLAS both
  run every session, and bf16 kernels are judged against the latter. Comparing
  a bf16 kernel to strict fp32 stacks the deck.
- **Logs are append-only.** Every run appends JSONL to `bench/results/`, and
  the charts are generated from those logs, never retyped.

## Design decisions

- **Why beating fp32 cuBLAS was possible.** I profiled the baselines with nsys
  before trusting any comparison. On sm_121, cuBLAS picks
  `cutlass_80_simt_sgemm_128x128_8x4_nn_align1` for fp32 and
  `cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4` for TF32. Both are
  sm_80 Ampere fallbacks; there's no sm_121-tuned path. That's the opening the
  bf16 tensor-core rung exploits, and it's why I report it. An unexplained win
  over a vendor library convinces nobody.
- **Built for what sm_121 actually is.** GB10 is *consumer* Blackwell: no
  `wgmma` (that's Hopper), no `tcgen05` (datacenter Blackwell), and 99 KB of
  shared memory per block. Published Hopper FlashAttention configs need 160+ KB
  and won't even launch here. So everything is built on `mma.sync.m16n8k16`,
  `ldmatrix`, and `cp.async`, tiles are sized against the 99 KB cap, and
  `make probe` checks the instruction set against ptxas instead of assuming it.
- **The attention kernel chains its two matmuls in registers.** The fragment
  layouts line up: two n-adjacent 16×8 QKᵀ accumulator tiles *are* the 16×16
  A-fragment the P@V `mma.sync` expects. After the online-softmax rescale, the
  probabilities never touch shared memory. I kept the shared-memory round trip
  as a compile-time alternative and verified the register path bit-identical
  against it before switching. Double-buffering K/V with `cp.async` was worth
  another +64%.
- **Tile sweeps are real tooling.** The sweep scripts rebuild each config,
  reject impossible ones up front (the 99 KB and 1024-thread limits) with the
  reason logged, verify at a small size, and only then time. The SGEMM sweep
  found BK=16. A 7-config attention sweep found Br=64/Bc=32, worth +13.6%
  because 40 KB of shared memory fits two resident blocks per SM where the
  64×64 default fits one.
- **Every kernel has a fallback.** The `float4` kernel falls back to the scalar
  2D-blocktile for shapes not divisible by 4, the tensor-core GEMM to the
  vectorized kernel for shapes not divisible by 16, and the tensor-core flash
  kernel to the fp32 flash kernel for ragged S. Every shape stays correct off
  the fast path, and `make run-ragged` exists to prove it.
- **What I didn't reach.** The best fp32 kernel stops at 77% of fp32 cuBLAS
  (rung 4 at ~59%). The profiler names the gap: cuBLAS's kernel runs a 4-stage
  `cp.async` pipeline and my fp32 kernels are single-buffered. That technique
  arrives in the tensor-core rung, applied to bf16 first. Likewise, the
  tensor-core GEMM's 2048³-vs-4096³ gap (38.2 vs 20.5 TFLOP/s) is the 64×64
  block tile paying DRAM re-fetch once the matrices outgrow the 24 MB L2. A
  128×128 warp-tiled block is the obvious next rung, and I'd rather name it
  than pretend the ladder is finished.

## The PyTorch custom op

`torch_ext/` wraps the fastest fp32 rung as a real operator,
`torch.ops.kernel_ladder.sgemm`. A microbenchmark speedup is a claim about a
kernel; an end-to-end speedup is a claim about a system, and I wanted both.

What it passes:

- the full correctness suite against `torch.matmul`, including non-contiguous
  views and `nn.Module` composition
- `torch.compile(fullgraph=True)` with no graph break, via a registered Meta
  kernel
- through the dispatcher it runs at 0.78× of fp32 `torch.matmul`, identical
  to the C++ harness ratio, so nothing is lost to binding overhead

What it shows inside a 4-layer MLP:

- 0.75× per-GEMM, yet **1.10× end-to-end** against the fp32 baseline
- Amdahl with p=0.56 caps the ceiling at 2.27× even if the GEMM were free
- **0.46× against stock PyTorch with TF32 on.** That's the honest bar for an
  fp32 kernel on this hardware, and the benchmark prints it.

```bash
cd torch_ext
python setup.py build_ext --inplace
python test_op.py          # correctness, composition, torch.compile, benchmark
python bench_in_model.py   # the kernel inside an MLP; Amdahl accounting
```

## Usage

```bash
make check          # preflight: driver, CUDA version, clocks, profiling perms
make probe          # ask ptxas which tensor-core instructions sm_121 supports
make build          # CMake configure + compile (ARCH=121)
make test-host      # CPU oracle + statistics unit tests, no GPU needed
make run            # SGEMM ladder at 4096³ (override with SIZE=...)
make run-small      # 512³, fast loop while developing
make run-ragged     # 1024x2048x512, catches M/N transposition bugs
make run-attn       # attention ladder (override with ATTN_SIZE=BxHxSxD)
make report         # SGEMM table + charts from bench/results/sgemm.jsonl
make report-attn    # attention tables + charts
make profile K=05_vectorized   # Nsight Compute on one kernel
make sanitize K=05_vectorized  # compute-sanitizer memcheck
make sass           # count load/store widths in SASS (LDG.E.128 after rung 5)
make sweep          # rung 4 tile-size sweep
```

GPU performance counters are admin-restricted by default
(`RmProfilingAdminOnly`). `scripts/profile.sh` uses a passwordless sudo rule
for `ncu` when one is installed, and warns otherwise.

## Repository layout

```
src/common/      CPU oracle, benchmark harness, statistics, GB10 constants
src/sgemm/       the SGEMM ladder (rungs 00–06 + cuBLAS fp32/TF32 baselines)
src/attention/   the attention ladder (naive → fused → flash → tensor cores)
torch_ext/       the PyTorch custom op + in-model benchmark
bench/           report generators; results/ holds JSONL logs and charts
scripts/         preflight check, ISA probe, profiling, tile sweeps, ceilings
tests/           host-only unit tests for the oracles and statistics
examples/        standalone one-warp mma.sync fragment-layout demo
```

## Hardware notes: GB10 / sm_121

There's no public GB10 architecture whitepaper, so these are numbers I
measured or verified locally.

| | |
|---|---|
| Machine | NVIDIA DGX Spark: GB10 Grace Blackwell, sm_121, 121 GB unified memory |
| DRAM bandwidth | 231 GB/s measured (spec sheet says 273; I use the measured number as the roofline slope) |
| fp32 (CUDA cores) | ~29.7 TFLOP/s, derived, not published |
| TF32 / bf16 tensor | 53.3 / 212.9 TFLOP/s measured; **FP8 equals the bf16 rate**, no 2× like datacenter parts |
| Ridge points | fp32 ~129 FLOP/byte, bf16 ~922; a naive GEMM starts at ~0.25 |
| Shared memory | **99 KB per block** (Hopper allows 228; Hopper flash configs won't launch) |
| Tensor-core ISA | `mma.sync m16n8k16`, `ldmatrix`, `cp.async`; **no `wgmma`, no `tcgen05`** |
| Clocks | can't be locked; ~2.4–2.5 GHz sustained under load, so I log clocks per run |
| Profiler gap 1 | no `dram__*` counters on this SoC; I substitute `lts__t_sectors_op_read*` (L2 traffic and misses) |
| Profiler gap 2 | no hmma-cycles metric; I count `sm__inst_executed_pipe_tensor.sum` (it matches the theoretical mma count exactly) and verify HMMA/LDSM/LDGSTS in SASS |
| cuBLAS | falls back to sm_80 CUTLASS kernels; no sm_121-tuned path |

## Requirements

- CUDA ≥ 12.9 (sm_121 needs PTX ISA 8.8; built and measured on 13.0)
- CMake ≥ 3.24
- Python 3.10+ with matplotlib and numpy for the reports
- For the custom op: a PyTorch build with sm_121 support (2.12.0+cu130 here)
- Nsight Compute / Nsight Systems for the profiling targets
