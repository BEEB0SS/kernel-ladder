// Benchmark harness: correctness gate, warmup, per-iteration event timing, clock sampling, JSONL logging.

#pragma once

#include "check.cuh"
#include "stats.hpp"
#include "cpu_reference.hpp"
#include "gb10.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <functional>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <array>
#include <memory>

namespace ladder {

struct GemmProblem {
    int   M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::string label() const {
        return std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    }
};

// fp32 and bf16 copies of A/B are prepared before timing so conversion is never in the timed region.
struct DeviceBuffers {
    float*           A_f32  = nullptr;
    float*           B_f32  = nullptr;
    float*           C      = nullptr;   // written by the kernel under test
    float*           C_init = nullptr;   // pristine copy of C, restored before each run
    __nv_bfloat16*   A_bf16 = nullptr;
    __nv_bfloat16*   B_bf16 = nullptr;
    std::size_t      bytes_A_f32 = 0, bytes_B_f32 = 0, bytes_C = 0;
};

struct GemmArgs {
    int M, N, K;
    float alpha, beta;
    const float*         A_f32;
    const float*         B_f32;
    const __nv_bfloat16* A_bf16;
    const __nv_bfloat16* B_bf16;
    float*               C;
};

enum class Precision { FP32, BF16_TF32 };

// A rung on the ladder.
struct KernelSpec {
    std::string name;          // short id, e.g. "02_smem_tiled"
    std::string description;   // one line: what changed vs the previous rung
    std::function<void(const GemmArgs&, cudaStream_t)> launch;
    Precision   precision  = Precision::FP32;

    // Per-kernel tolerance: fp32 matches the double oracle to ~1e-4; bf16 legitimately lands ~1e-2.
    double      tolerance  = 2e-4;

    bool        is_baseline = false;   // cuBLAS -- the number to beat
    bool        implemented = true;    // disabled kernels are skipped rather than failed
};

// GB10 cannot lock clocks, so drift is sampled and reported with every result.
struct ClockSample {
    double sm_clock_mhz  = 0.0;
    double mem_clock_mhz = 0.0;
    double power_w       = 0.0;
    double temp_c        = 0.0;
    bool   valid         = false;
};

inline ClockSample sample_clocks() {
    ClockSample s;
    // NVML is partially blind on Spark; nvidia-smi clocks/power/temp do work.
    std::FILE* p = popen(
        "nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return s;
    char buf[256] = {0};
    if (std::fgets(buf, sizeof(buf), p)) {
        // Spark reports clocks.mem as "[N/A]"; parse per-field, not whole-line.
        double vals[4] = {0, 0, 0, 0};
        int idx = 0;
        for (char* tok = std::strtok(buf, ","); tok && idx < 4;
             tok = std::strtok(nullptr, ","), ++idx) {
            vals[idx] = std::atof(tok);   // "[N/A]" parses as 0
        }
        if (vals[0] > 0) {
            s.sm_clock_mhz = vals[0]; s.mem_clock_mhz = vals[1];
            s.power_w = vals[2]; s.temp_c = vals[3]; s.valid = true;
        }
    }
    pclose(p);
    return s;
}

struct BenchResult {
    std::string   kernel_name;
    std::string   description;
    GemmProblem   problem;
    TimingStats   timing;
    CompareResult correctness;
    bool          skipped      = false;
    double        gflops_p50   = 0.0;
    double        gflops_min   = 0.0;
    double        pct_of_peak  = 0.0;   // vs the relevant compute ceiling
    double        speedup_vs_baseline = 0.0;
    ClockSample   clocks_before, clocks_after;
    std::string   precision_label;
};

struct BenchConfig {
    int  warmup_iters  = 25;
    int  measure_iters = 100;
    bool verify        = true;

    // Idle gaps drop clocks and fatten the tail; enable only when investigating thermals.
    bool sleep_between = false;
};

class Benchmarker {
public:
    Benchmarker(const GemmProblem& prob, const BenchConfig& cfg)
        : prob_(prob), cfg_(cfg) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreate(&ev_start_));
        CUDA_CHECK(cudaEventCreate(&ev_stop_));
    }

    ~Benchmarker() {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
        cudaStreamDestroy(stream_);
    }

    // CPU oracle is O(M*N*K); computed once per problem size and reused for every rung.
    void prepare_reference(const std::vector<float>& hA,
                           const std::vector<float>& hB,
                           const std::vector<float>& hC_init) {
        reference_ = hC_init;
        std::printf("  computing CPU reference (%s)... ", prob_.label().c_str());
        std::fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        sgemm_cpu(prob_.M, prob_.N, prob_.K, prob_.alpha,
                  hA.data(), hB.data(), prob_.beta, reference_.data());
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("%.1fs\n",
            std::chrono::duration<double>(t1 - t0).count());
        have_reference_ = true;
    }

    BenchResult run(const KernelSpec& spec, const DeviceBuffers& buf) {
        BenchResult r;
        r.kernel_name = spec.name;
        r.description = spec.description;
        r.problem     = prob_;
        r.precision_label = (spec.precision == Precision::FP32) ? "fp32" : "bf16/tf32";

        if (!spec.implemented) {
            r.skipped = true;
            return r;
        }

        GemmArgs args{prob_.M, prob_.N, prob_.K, prob_.alpha, prob_.beta,
                      buf.A_f32, buf.B_f32, buf.A_bf16, buf.B_bf16, buf.C};

        // Restore C first: with beta != 0, reruns would otherwise accumulate on prior output.
        if (cfg_.verify && have_reference_) {
            CUDA_CHECK(cudaMemcpy(buf.C, buf.C_init, buf.bytes_C,
                                  cudaMemcpyDeviceToDevice));
            spec.launch(args, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            CUDA_CHECK_KERNEL();

            std::vector<float> host_out(static_cast<std::size_t>(prob_.M) * prob_.N);
            CUDA_CHECK(cudaMemcpy(host_out.data(), buf.C, buf.bytes_C,
                                  cudaMemcpyDeviceToHost));
            // Scale floor sqrt(K): the natural |C| magnitude for N(0,1) inputs.
            r.correctness = compare(host_out, reference_, spec.tolerance,
                                    std::sqrt(static_cast<double>(prob_.K)));

            if (!r.correctness.passed) {
                return r;
            }
        } else {
            r.correctness.passed = true;   // verification explicitly disabled
        }

        for (int i = 0; i < cfg_.warmup_iters; ++i) {
            spec.launch(args, stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK_KERNEL();

        r.clocks_before = sample_clocks();

        std::vector<double> samples;
        samples.reserve(cfg_.measure_iters);

        for (int i = 0; i < cfg_.measure_iters; ++i) {
            CUDA_CHECK(cudaEventRecord(ev_start_, stream_));
            spec.launch(args, stream_);
            CUDA_CHECK(cudaEventRecord(ev_stop_, stream_));
            CUDA_CHECK(cudaEventSynchronize(ev_stop_));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start_, ev_stop_));
            samples.push_back(static_cast<double>(ms));
        }

        r.clocks_after = sample_clocks();
        r.timing = summarize(samples);

        const double flops = gemm_flops(prob_.M, prob_.N, prob_.K);
        r.gflops_p50 = gflops(flops, r.timing.p50_ms);
        r.gflops_min = gflops(flops, r.timing.min_ms);

        const double ceiling_gflops =
            (spec.precision == Precision::FP32)
                ? gb10::kFp32PeakTFLOPS   * 1000.0
                : gb10::kBf16TensorTFLOPS * 1000.0;
        r.pct_of_peak = 100.0 * r.gflops_p50 / ceiling_gflops;

        if (spec.is_baseline) baseline_gflops_ = r.gflops_p50;
        if (baseline_gflops_ > 0.0)
            r.speedup_vs_baseline = r.gflops_p50 / baseline_gflops_;

        return r;
    }

    cudaStream_t stream() const { return stream_; }

private:
    GemmProblem        prob_;
    BenchConfig        cfg_;
    cudaStream_t       stream_{};
    cudaEvent_t        ev_start_{}, ev_stop_{};
    std::vector<float> reference_;
    bool               have_reference_ = false;
    double             baseline_gflops_ = 0.0;
};

inline void print_result_header() {
    std::printf("\n%-22s %-10s %9s %9s %9s %8s %9s %7s  %s\n",
                "kernel", "precision", "p50 (ms)", "min (ms)", "GFLOP/s",
                "%peak", "vs cuBLAS", "cv", "status");
    std::printf("%s\n", std::string(118, '-').c_str());
}

inline void print_result(const BenchResult& r) {
    if (r.skipped) {
        std::printf("%-22s %-10s %9s %9s %9s %8s %9s %7s  %s\n",
            r.kernel_name.c_str(), "-", "-", "-", "-", "-", "-", "-",
            "not implemented (skipped)");
        return;
    }
    if (!r.correctness.passed) {
        std::printf("%-22s %-10s %9s %9s %9s %8s %9s %7s  WRONG: max rel err %.3g",
            r.kernel_name.c_str(), r.precision_label.c_str(),
            "-", "-", "-", "-", "-", "-", r.correctness.max_rel_error);
        if (r.correctness.first_bad_index >= 0) {
            std::printf(" (first bad idx %d: got %.6g want %.6g)",
                r.correctness.first_bad_index,
                r.correctness.first_bad_got, r.correctness.first_bad_want);
        }
        std::printf("\n");
        return;
    }
    const char* health = (r.timing.cv() > 0.05) ? "  [NOISY: cv>5%, rerun on a quiet box]" : "";
    std::printf("%-22s %-10s %9.3f %9.3f %9.1f %7.1f%% %8.2fx %7.3f  ok%s\n",
        r.kernel_name.c_str(), r.precision_label.c_str(),
        r.timing.p50_ms, r.timing.min_ms, r.gflops_p50,
        r.pct_of_peak, r.speedup_vs_baseline, r.timing.cv(), health);
}

// Append-only JSONL: one JSON object per result.
inline void append_jsonl(const std::string& path, const BenchResult& r) {
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "warning: could not open %s for append\n", path.c_str());
        return;
    }
    const std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    std::fprintf(f,
        "{\"timestamp\":\"%sZ\",\"kernel\":\"%s\",\"description\":\"%s\","
        "\"M\":%d,\"N\":%d,\"K\":%d,\"precision\":\"%s\","
        "\"skipped\":%s,\"correct\":%s,\"max_rel_error\":%.6g,"
        "\"p50_ms\":%.6f,\"min_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,"
        "\"mean_ms\":%.6f,\"stddev_ms\":%.6f,\"cv\":%.6f,\"n\":%d,"
        "\"gflops_p50\":%.3f,\"gflops_min\":%.3f,\"pct_of_peak\":%.3f,"
        "\"speedup_vs_cublas\":%.4f,"
        "\"sm_clock_before_mhz\":%.1f,\"sm_clock_after_mhz\":%.1f,"
        "\"power_w_after\":%.1f,\"temp_c_after\":%.1f}\n",
        ts, r.kernel_name.c_str(), r.description.c_str(),
        r.problem.M, r.problem.N, r.problem.K, r.precision_label.c_str(),
        r.skipped ? "true" : "false",
        r.correctness.passed ? "true" : "false", r.correctness.max_rel_error,
        r.timing.p50_ms, r.timing.min_ms, r.timing.p90_ms, r.timing.p99_ms,
        r.timing.mean_ms, r.timing.stddev_ms, r.timing.cv(), r.timing.n,
        r.gflops_p50, r.gflops_min, r.pct_of_peak, r.speedup_vs_baseline,
        r.clocks_before.sm_clock_mhz, r.clocks_after.sm_clock_mhz,
        r.clocks_after.power_w, r.clocks_after.temp_c);
    std::fclose(f);
}

}  // namespace ladder
