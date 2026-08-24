// SGEMM ladder driver: allocates, converts, verifies, benchmarks, logs.

#include "kernels.cuh"
#include "../common/check.cuh"
#include "../common/harness.cuh"

#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <sstream>

using namespace ladder;

__global__ void convert_f32_to_bf16(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst,
                                    std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

struct Options {
    GemmProblem prob;
    BenchConfig cfg;
    std::string results_path = "bench/results/sgemm.jsonl";
    std::set<std::string> only;     // empty = run everything
    bool list_only = false;
};

static void print_usage() {
    std::printf(
"kernel-ladder — SGEMM optimization ladder for DGX Spark (GB10 / sm_121)\n\n"
"Usage: ladder [options]\n\n"
"  --size N               square problem, M=N=K=N          (default 4096)\n"
"  --size MxNxK           explicit non-square problem\n"
"                         ALWAYS test a non-square size before believing a rung:\n"
"                         M/N transposition bugs are invisible when M==N.\n"
"  --warmup N             discarded iterations             (default 25)\n"
"  --iters N              measured iterations              (default 100)\n"
"  --no-verify            skip the correctness gate (fast iteration; never\n"
"                         report a number produced with this flag)\n"
"  --only a,b,c           run only these kernels by name\n"
"  --list                 list kernel names and exit\n"
"  --out PATH             JSONL results file  (default bench/results/sgemm.jsonl)\n"
"  --alpha F / --beta F   GEMM scalars                     (default 1.0 / 0.0)\n"
"  -h, --help             this message\n\n"
"Examples:\n"
"  ./build/ladder --size 4096\n"
"  ./build/ladder --size 1024x2048x512      # catches M/N mixups\n"
"  ./build/ladder --only 00_naive,01_coalesced --size 2048\n");
}

static bool parse_size(const std::string& s, GemmProblem& p) {
    if (s.find('x') == std::string::npos) {
        const int n = std::atoi(s.c_str());
        if (n <= 0) return false;
        p.M = p.N = p.K = n;
        return true;
    }
    std::stringstream ss(s);
    std::string tok;
    int dims[3] = {0, 0, 0};
    int i = 0;
    while (std::getline(ss, tok, 'x') && i < 3) dims[i++] = std::atoi(tok.c_str());
    if (i != 3 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) return false;
    p.M = dims[0]; p.N = dims[1]; p.K = dims[2];
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
        else if (a == "--size")           { if (!parse_size(next(), o.prob)) { std::fprintf(stderr, "bad --size\n"); std::exit(1); } }
        else if (a == "--warmup")         { o.cfg.warmup_iters  = std::atoi(next().c_str()); }
        else if (a == "--iters")          { o.cfg.measure_iters = std::atoi(next().c_str()); }
        else if (a == "--no-verify")      { o.cfg.verify = false; }
        else if (a == "--out")            { o.results_path = next(); }
        else if (a == "--alpha")          { o.prob.alpha = static_cast<float>(std::atof(next().c_str())); }
        else if (a == "--beta")           { o.prob.beta  = static_cast<float>(std::atof(next().c_str())); }
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
    int clock_khz = 0;  // cudaDeviceProp::clockRate was removed in CUDA 13
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
    std::printf("  L2 cache %d MB, clock %.2f GHz\n",
                p.l2CacheSize / (1024 * 1024), clock_khz / 1e6);

    if (p.major == 12 && p.minor == 1) {
        std::printf("  -> sm_121 confirmed. No wgmma, no tcgen05; mma.sync is the tensor-core path.\n");
    } else {
        std::printf("  -> WARNING: expected sm_121 (GB10). The tile sizes and the\n"
                    "     99KB shared-memory assumptions in this repo are tuned for\n"
                    "     GB10 and may be wrong here.\n");
    }

    const ClockSample c = sample_clocks();
    if (c.valid) {
        std::printf("  clocks now: SM %.0f MHz, mem %.0f MHz, %.1f W, %.0f C\n",
                    c.sm_clock_mhz, c.mem_clock_mhz, c.power_w, c.temp_c);
        // GB10 quirk: SM clock can pin at 721 MHz with no throttle reason; only a power cycle fixes it.
        if (c.sm_clock_mhz > 0 && c.sm_clock_mhz < 900) {
            std::printf("\n  *** WARNING: SM clock is %.0f MHz, far below the ~2.4 GHz\n"
                        "      expected. GB10 has a known state where it pins at 721 MHz\n"
                        "      under load with no throttle reason and cannot be unstuck\n"
                        "      with nvidia-smi -lgc. POWER CYCLE THE BOX before recording\n"
                        "      any results from this session.\n\n", c.sm_clock_mhz);
        }
    }

    std::size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("  cudaMemGetInfo: %.1f / %.1f GB free\n",
                free_b / 1e9, total_b / 1e9);
    std::printf("  (on Spark this UNDER-reports: the Linux page cache holds\n"
                "   reclaimable memory from the same unified pool. If you hit a\n"
                "   surprising OOM, run: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches')\n");
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    auto ladder_specs = build_ladder();

    if (opt.list_only) {
        std::printf("\nkernels:\n");
        for (const auto& k : ladder_specs)
            std::printf("  %-18s %-9s %s%s\n", k.name.c_str(),
                        k.precision == Precision::FP32 ? "[fp32]" : "[bf16]",
                        k.description.c_str(),
                        k.implemented ? "" : "   (NOT IMPLEMENTED)");
        std::printf("\n");
        return 0;
    }

    print_device_banner();

    const GemmProblem& P = opt.prob;
    std::printf("\n== problem ==\n");
    std::printf("  M=%d N=%d K=%d  alpha=%.3f beta=%.3f\n", P.M, P.N, P.K, P.alpha, P.beta);
    std::printf("  %.2f GFLOP per call, arithmetic intensity %.1f FLOP/byte\n",
                gemm_flops(P.M, P.N, P.K) / 1e9,
                arithmetic_intensity(P.M, P.N, P.K));
    std::printf("  fp32 ridge point is %.0f FLOP/byte -> this problem is %s\n",
                gb10::kRidgePointFp32,
                arithmetic_intensity(P.M, P.N, P.K) > gb10::kRidgePointFp32
                    ? "COMPUTE-bound (good: optimization can pay off)"
                    : "MEMORY-bound (tiling cannot help; shrink the output or fuse)");
    std::printf("  warmup=%d iters=%d verify=%s\n",
                opt.cfg.warmup_iters, opt.cfg.measure_iters,
                opt.cfg.verify ? "yes" : "NO (numbers are not reportable)");

    // Host buffers must be pinned: pageable H2D copies are pathologically slow on Spark (ARM SMMU path).
    const std::size_t nA = static_cast<std::size_t>(P.M) * P.K;
    const std::size_t nB = static_cast<std::size_t>(P.K) * P.N;
    const std::size_t nC = static_cast<std::size_t>(P.M) * P.N;

    std::vector<float> hA = make_matrix(P.M, P.K, 1337);
    std::vector<float> hB = make_matrix(P.K, P.N, 7331);
    std::vector<float> hC = make_matrix(P.M, P.N, 4242);

    float *pA = nullptr, *pB = nullptr, *pC = nullptr;
    CUDA_CHECK(cudaMallocHost(&pA, nA * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pB, nB * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&pC, nC * sizeof(float)));
    std::memcpy(pA, hA.data(), nA * sizeof(float));
    std::memcpy(pB, hB.data(), nB * sizeof(float));
    std::memcpy(pC, hC.data(), nC * sizeof(float));

    DeviceBuffers buf;
    buf.bytes_A_f32 = nA * sizeof(float);
    buf.bytes_B_f32 = nB * sizeof(float);
    buf.bytes_C     = nC * sizeof(float);

    CUDA_CHECK(cudaMalloc(&buf.A_f32,  buf.bytes_A_f32));
    CUDA_CHECK(cudaMalloc(&buf.B_f32,  buf.bytes_B_f32));
    CUDA_CHECK(cudaMalloc(&buf.C,      buf.bytes_C));
    CUDA_CHECK(cudaMalloc(&buf.C_init, buf.bytes_C));
    CUDA_CHECK(cudaMalloc(&buf.A_bf16, nA * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&buf.B_bf16, nB * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(buf.A_f32,  pA, buf.bytes_A_f32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.B_f32,  pB, buf.bytes_B_f32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.C_init, pC, buf.bytes_C,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.C,      pC, buf.bytes_C,     cudaMemcpyHostToDevice));

    // bf16 conversion happens once at setup, never inside a timed region.
    {
        const int threads = 256;
        convert_f32_to_bf16<<<ceil_div(static_cast<int>(nA), threads), threads>>>(
            buf.A_f32, buf.A_bf16, nA);
        convert_f32_to_bf16<<<ceil_div(static_cast<int>(nB), threads), threads>>>(
            buf.B_f32, buf.B_bf16, nB);
        CUDA_CHECK_KERNEL();
    }

    cublas_init();

    Benchmarker bench(P, opt.cfg);
    if (opt.cfg.verify) {
        std::printf("\n== correctness oracle ==\n");
        const double est_s = 2.0 * static_cast<double>(P.M) * P.N * P.K / 2e9;
        if (est_s > 30.0) {
            std::printf("  heads up: this is ~%.0fs of single-threaded CPU work.\n"
                        "  Develop at --size 512 or 1024, then verify once at full size.\n", est_s);
        }
        bench.prepare_reference(hA, hB, hC);
    }

    std::printf("\n== ladder ==");
    print_result_header();

    int implemented = 0, correct = 0;
    for (const auto& spec : ladder_specs) {
        if (!opt.only.empty() && opt.only.count(spec.name) == 0) continue;
        BenchResult r = bench.run(spec, buf);
        print_result(r);
        append_jsonl(opt.results_path, r);
        if (!r.skipped) { ++implemented; if (r.correctness.passed) ++correct; }
    }

    std::printf("\n%d kernels run, %d correct. results appended to %s\n",
                implemented, correct, opt.results_path.c_str());
    std::printf("next: python3 bench/report.py %s\n\n", opt.results_path.c_str());

    cublas_destroy();
    CUDA_CHECK(cudaFree(buf.A_f32));  CUDA_CHECK(cudaFree(buf.B_f32));
    CUDA_CHECK(cudaFree(buf.C));      CUDA_CHECK(cudaFree(buf.C_init));
    CUDA_CHECK(cudaFree(buf.A_bf16)); CUDA_CHECK(cudaFree(buf.B_bf16));
    CUDA_CHECK(cudaFreeHost(pA));     CUDA_CHECK(cudaFreeHost(pB));
    CUDA_CHECK(cudaFreeHost(pC));
    return (implemented > 0 && correct == implemented) ? 0 : 1;
}
