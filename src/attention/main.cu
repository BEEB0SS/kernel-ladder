// main.cu — driver for the attention ladder: allocate, convert, verify, benchmark, log.

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <sstream>

using namespace ladder;

// bf16 conversion runs once at setup, never inside a timed region.
__global__ void attn_f32_to_bf16(const float* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst,
                                 std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

struct Options {
    AttnProblem prob;
    BenchConfig cfg;
    std::string results_path = "bench/results/attention.jsonl";
    std::set<std::string> only;
    bool list_only = false;
};

static void print_usage() {
    std::printf(
"kernel-ladder — fused attention ladder for DGX Spark (GB10 / sm_121)\n\n"
"Usage: attention [options]\n\n"
"  --size BxHxSxD         batch x heads x seq_len x head_dim   (default 4x8x1024x64)\n"
"  --seq N                change only the sequence length\n"
"  --causal               apply the causal mask (decoder-style attention)\n"
"  --warmup N             discarded iterations                 (default 25)\n"
"  --iters N              measured iterations                  (default 100)\n"
"  --no-verify            skip the correctness gate (fast iteration; never\n"
"                         report a number produced with this flag)\n"
"  --only a,b,c           run only these kernels by name\n"
"  --list                 list kernel names and exit\n"
"  --out PATH             JSONL results  (default bench/results/attention.jsonl)\n"
"  -h, --help             this message\n\n"
"Examples:\n"
"  ./build/attention --size 4x8x1024x64\n"
"  ./build/attention --seq 256 --only 02_flash_tiled      # fast dev loop\n"
"  ./build/attention --size 2x5x384x64 --causal           # ragged, catches index bugs\n"
"  ./build/attention --size 4x8x8192x64 --only 02_flash_tiled\n"
"                                        # a size rung 0 cannot allocate\n\n"
"NOTE ON SHAPES: always test with B, H, S and D all DIFFERENT before believing\n"
"a rung. A kernel that indexes heads where it meant batches is perfectly\n"
"correct whenever B == H.\n");
}

static bool parse_shape(const std::string& s, AttnProblem& p) {
    std::stringstream ss(s);
    std::string tok;
    int dims[4] = {0, 0, 0, 0};
    int i = 0;
    while (std::getline(ss, tok, 'x') && i < 4) dims[i++] = std::atoi(tok.c_str());
    if (i != 4) return false;
    for (int k = 0; k < 4; ++k) if (dims[k] <= 0) return false;
    p.batch = dims[0]; p.heads = dims[1]; p.seq_len = dims[2]; p.head_dim = dims[3];
    return true;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help")   { print_usage(); std::exit(0); }
        else if (a == "--size")           { if (!parse_shape(next(), o.prob)) { std::fprintf(stderr, "bad --size, want BxHxSxD\n"); std::exit(1); } }
        else if (a == "--seq")            { o.prob.seq_len = std::atoi(next().c_str()); }
        else if (a == "--causal")         { o.prob.causal = true; }
        else if (a == "--warmup")         { o.cfg.warmup_iters  = std::atoi(next().c_str()); }
        else if (a == "--iters")          { o.cfg.measure_iters = std::atoi(next().c_str()); }
        else if (a == "--no-verify")      { o.cfg.verify = false; }
        else if (a == "--out")            { o.results_path = next(); }
        else if (a == "--list")           { o.list_only = true; }
        else if (a == "--only") {
            std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) o.only.insert(t);
        } else {
            std::fprintf(stderr, "unknown option: %s\n\n", a.c_str());
            print_usage(); std::exit(1);
        }
    }
    return o;
}

static void print_device_banner() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    std::printf("\n== device ==\n");
    std::printf("  %s  (compute capability %d.%d)\n", p.name, p.major, p.minor);
    std::printf("  %d SMs, %.1f GB global, %d KB shared/block max\n",
                p.multiProcessorCount,
                p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                static_cast<int>(p.sharedMemPerBlockOptin / 1024));

    if (p.major == 12 && p.minor == 1) {
        std::printf("  -> sm_121 confirmed. No wgmma, no tcgen05, 99KB smem/block.\n"
                    "     FlashAttention tile sizes from Hopper repos will not fit.\n"
                    "     mma.sync + ldmatrix + cp.async are the available primitives.\n");
    }

    const ClockSample c = sample_clocks();
    if (c.valid) {
        std::printf("  clocks now: SM %.0f MHz, mem %.0f MHz, %.1f W, %.0f C\n",
                    c.sm_clock_mhz, c.mem_clock_mhz, c.power_w, c.temp_c);
        if (c.sm_clock_mhz > 0 && c.sm_clock_mhz < 900) {
            std::printf("\n  *** WARNING: SM clock is %.0f MHz. GB10 has a known state where\n"
                        "      it pins at 721 MHz under load and cannot be unstuck with\n"
                        "      nvidia-smi -lgc. POWER CYCLE before recording results.\n\n",
                        c.sm_clock_mhz);
        }
    }
}

static void print_problem_banner(const AttnProblem& P, bool need_scratch) {
    const double flops   = attention_flops(P.batch, P.heads, P.seq_len, P.head_dim, P.causal);
    const double ai_fused = attention_intensity_ideal(P.batch, P.heads, P.seq_len,
                                                      P.head_dim, P.causal, 4);
    const double ai_naive = attention_intensity_naive(P.batch, P.heads, P.seq_len,
                                                      P.head_dim, P.causal, 4);

    std::printf("\n== problem ==\n");
    std::printf("  B=%d H=%d S=%d D=%d  causal=%s  scale=%.6f\n",
                P.batch, P.heads, P.seq_len, P.head_dim,
                P.causal ? "yes" : "no", P.scale());
    std::printf("  %.2f GFLOP per call (4*B*H*S*S*D%s)\n",
                flops / 1e9, P.causal ? ", halved by the causal mask" : "");

    std::printf("\n  arithmetic intensity, fp32:\n");
    std::printf("    fused (rungs 2-3)   %8.1f FLOP/byte   = S/4, independent of D\n", ai_fused);
    std::printf("    naive (rungs 0-1)   %8.1f FLOP/byte   -> capped at D/4 = %.1f, flat in S\n",
                ai_naive, P.head_dim / 4.0);
    std::printf("    GB10 fp32 ridge point is %.0f FLOP/byte\n", gb10::kRidgePointFp32);
    std::printf("    -> a fused fp32 kernel here is %s\n",
                ai_fused > gb10::kRidgePointFp32
                    ? "COMPUTE-bound (arithmetic is the target)"
                    : "MEMORY-bound (data movement is the target; more math per byte will not help)");
    std::printf("    -> the naive path is %s, and cannot escape it at any S\n",
                ai_naive > gb10::kRidgePointFp32 ? "compute-bound" : "MEMORY-bound");

    std::printf("\n  the O(S^2) score matrix:\n");
    std::printf("    B*H*S*S fp32 = %.3f GB   %s\n",
                score_matrix_bytes(P.batch, P.heads, P.seq_len) / 1e9,
                need_scratch ? "(allocating -- a selected rung needs it)"
                             : "(NOT allocating -- no selected rung needs it)");
    std::printf("    at S=%d it would be %.3f GB; it grows as S^2\n",
                P.seq_len * 2,
                score_matrix_bytes(P.batch, P.heads, P.seq_len * 2) / 1e9);

    std::printf("\n  reference semantics (what the oracle computes):\n");
    std::printf("    torch.nn.functional.scaled_dot_product_attention(\n");
    std::printf("        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=%s)\n",
                P.causal ? "True" : "False");
    std::printf("    with q,k,v of shape (%d, %d, %d, %d) and dtype float32,\n",
                P.batch, P.heads, P.seq_len, P.head_dim);
    std::printf("    i.e. the (B, H, S, D) layout torch gives you after\n"
                "    x.view(B, S, H, D).transpose(1, 2). Same 1/sqrt(D) scale.\n");
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    auto specs = build_attn_ladder();

    if (opt.list_only) {
        std::printf("\nattention kernels:\n");
        for (const auto& k : specs)
            std::printf("  %-22s %-7s %s%s\n", k.name.c_str(),
                        k.precision == Precision::FP32 ? "[fp32]" : "[bf16]",
                        k.description.c_str(),
                        k.implemented ? "" : "   (NOT IMPLEMENTED)");
        std::printf("\n");
        return 0;
    }

    // The O(S^2) score workspace is allocated only if a selected rung needs it.
    bool need_scratch = false;
    for (const auto& s : specs) {
        if (!opt.only.empty() && opt.only.count(s.name) == 0) continue;
        if (s.implemented && s.needs_scratch) need_scratch = true;
    }

    print_device_banner();
    const AttnProblem& P = opt.prob;
    print_problem_banner(P, need_scratch);
    std::printf("\n  warmup=%d iters=%d verify=%s\n",
                opt.cfg.warmup_iters, opt.cfg.measure_iters,
                opt.cfg.verify ? "yes" : "NO (numbers are not reportable)");

    // Host staging buffers must be pinned: pageable H2D on Spark is ~5x slower.
    const std::size_t n = P.qkv_elems();
    std::vector<float> hQ = make_qkv(P.batch, P.heads, P.seq_len, P.head_dim, 1337);
    std::vector<float> hK = make_qkv(P.batch, P.heads, P.seq_len, P.head_dim, 7331);
    std::vector<float> hV = make_qkv(P.batch, P.heads, P.seq_len, P.head_dim, 4242);

    float *pQ = nullptr, *pK = nullptr, *pV = nullptr;
    CUDA_CHECK(cudaMallocHost(&pQ, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pK, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pV, n * sizeof(float)));
    std::memcpy(pQ, hQ.data(), n * sizeof(float));
    std::memcpy(pK, hK.data(), n * sizeof(float));
    std::memcpy(pV, hV.data(), n * sizeof(float));

    AttnBuffers buf;
    buf.bytes_qkv = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&buf.Q, buf.bytes_qkv));
    CUDA_CHECK(cudaMalloc(&buf.K, buf.bytes_qkv));
    CUDA_CHECK(cudaMalloc(&buf.V, buf.bytes_qkv));
    CUDA_CHECK(cudaMalloc(&buf.O, buf.bytes_qkv));
    CUDA_CHECK(cudaMalloc(&buf.Q_bf16, n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&buf.K_bf16, n * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&buf.V_bf16, n * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(buf.Q, pQ, buf.bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.K, pK, buf.bytes_qkv, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.V, pV, buf.bytes_qkv, cudaMemcpyHostToDevice));

    if (need_scratch) {
        buf.scratch_elems = P.score_elems();
        const std::size_t bytes = buf.scratch_elems * sizeof(float);
        std::size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        if (bytes > free_b) {
            std::fprintf(stderr,
                "\nCannot allocate the %.2f GB B*H*S*S score matrix (%.2f GB free).\n"
                "This is not a bug -- it is the naive path's O(S^2) score matrix -- the\n"
                "wall FlashAttention removes. Either drop S,\n"
                "or run the kernels that do not need it:\n"
                "    ./build/attention --size %dx%dx%dx%d --only 02_flash_tiled\n\n",
                bytes / 1e9, free_b / 1e9,
                P.batch, P.heads, P.seq_len, P.head_dim);
            return 1;
        }
        CUDA_CHECK(cudaMalloc(&buf.scratch, bytes));
    }

    {
        const int threads = 256;
        const int blocks  = ceil_div(static_cast<int>(n), threads);
        attn_f32_to_bf16<<<blocks, threads>>>(buf.Q, buf.Q_bf16, n);
        attn_f32_to_bf16<<<blocks, threads>>>(buf.K, buf.K_bf16, n);
        attn_f32_to_bf16<<<blocks, threads>>>(buf.V, buf.V_bf16, n);
        CUDA_CHECK_KERNEL();
    }

    AttnBenchmarker bench(P, opt.cfg);
    if (opt.cfg.verify) {
        std::printf("\n== correctness oracle ==\n");
        // Estimate assumes the single-threaded oracle runs at ~2 GFLOP/s.
        const double est_s = attention_flops(P.batch, P.heads, P.seq_len,
                                             P.head_dim, P.causal) / 2e9;
        if (est_s > 30.0) {
            std::printf("  heads up: this is ~%.0fs of single-threaded CPU work.\n"
                        "  Develop at --seq 256, then verify once at full size.\n", est_s);
        }
        bench.prepare_reference(hQ, hK, hV);
    }

    std::printf("\n== ladder ==");
    print_attn_header();

    int implemented = 0, correct = 0;
    for (const auto& spec : specs) {
        if (!opt.only.empty() && opt.only.count(spec.name) == 0) continue;
        AttnBenchResult r = bench.run(spec, buf);
        print_attn_result(r);
        append_attn_jsonl(opt.results_path, r);
        if (!r.skipped) { ++implemented; if (r.correctness.passed) ++correct; }
    }

    std::printf("\n%d kernels run, %d correct. results appended to %s\n",
                implemented, correct, opt.results_path.c_str());
    std::printf("next: python3 bench/report_attention.py %s\n\n", opt.results_path.c_str());

    CUDA_CHECK(cudaFree(buf.Q)); CUDA_CHECK(cudaFree(buf.K));
    CUDA_CHECK(cudaFree(buf.V)); CUDA_CHECK(cudaFree(buf.O));
    CUDA_CHECK(cudaFree(buf.Q_bf16)); CUDA_CHECK(cudaFree(buf.K_bf16));
    CUDA_CHECK(cudaFree(buf.V_bf16));
    if (buf.scratch) CUDA_CHECK(cudaFree(buf.scratch));
    CUDA_CHECK(cudaFreeHost(pQ)); CUDA_CHECK(cudaFreeHost(pK));
    CUDA_CHECK(cudaFreeHost(pV));
    return (implemented > 0 && correct == implemented) ? 0 : 1;
}
