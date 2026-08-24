// Attention benchmark harness: problem shape, buffers, and the correctness-gated
// benchmarker (shared machinery comes from src/common/harness.cuh).

#pragma once

#include "attention_reference.hpp"
#include "../common/harness.cuh"     // ClockSample, BenchConfig, Precision, stats
#include "../common/check.cuh"
#include "../common/gb10.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <functional>
#include <string>
#include <vector>
#include <cstdio>
#include <ctime>
#include <algorithm>

namespace ladder {

// Defaults are small so the CPU oracle stays fast; not a shape to report results at.
struct AttnProblem {
    int  batch    = 4;
    int  heads    = 8;
    int  seq_len  = 1024;
    int  head_dim = 64;
    bool causal   = false;

    // Single definition of 1/sqrt(D), shared by oracle, harness, and kernels.
    float scale() const { return attention_scale(head_dim); }

    std::size_t qkv_elems() const {
        return static_cast<std::size_t>(batch) * heads * seq_len * head_dim;
    }
    // O(S^2) score workspace; only scratch-using rungs allocate it.
    std::size_t score_elems() const {
        return static_cast<std::size_t>(batch) * heads * seq_len * seq_len;
    }

    std::string label() const {
        return "B" + std::to_string(batch) + "H" + std::to_string(heads) +
               "S" + std::to_string(seq_len) + "D" + std::to_string(head_dim) +
               (causal ? "-causal" : "");
    }
};

// bf16 copies are converted once at startup so no timed region measures conversion.
struct AttnBuffers {
    float*         Q      = nullptr;
    float*         K      = nullptr;
    float*         V      = nullptr;
    float*         O      = nullptr;
    __nv_bfloat16* Q_bf16 = nullptr;
    __nv_bfloat16* K_bf16 = nullptr;
    __nv_bfloat16* V_bf16 = nullptr;

    float*         scratch       = nullptr;   // B*H*S*S, may be null
    std::size_t    scratch_elems = 0;

    std::size_t    bytes_qkv = 0;             // one of Q/K/V/O, in fp32
};

struct AttnArgs {
    int   B, H, S, D;
    float scale;
    bool  causal;

    const float*         Q;
    const float*         K;
    const float*         V;
    const __nv_bfloat16* Q_bf16;
    const __nv_bfloat16* K_bf16;
    const __nv_bfloat16* V_bf16;

    float*      O;
    float*      scratch;        // B*H*S*S workspace, or nullptr
    std::size_t scratch_elems;
};

// A rung on the attention ladder.
struct AttnKernelSpec {
    std::string name;
    std::string description;
    std::function<void(const AttnArgs&, cudaStream_t)> launch;
    Precision   precision = Precision::FP32;

    // fp32 rungs must match the double-accumulated oracle to ~1e-4; bf16 lands near 1e-2.
    double tolerance = 2e-4;

    bool needs_scratch = false;   // does this rung want the B*H*S*S workspace?
    bool implemented   = true;    // disabled kernels are skipped rather than failed
};

struct AttnBenchResult {
    std::string   kernel_name;
    std::string   description;
    AttnProblem   problem;
    TimingStats   timing;
    CompareResult correctness;
    bool          skipped = false;

    double gflops_p50  = 0.0;
    double gflops_min  = 0.0;
    double pct_of_peak = 0.0;   // vs the achievable roofline ceiling
    double achieved_gbs = 0.0;  // effective DRAM bandwidth, ideal-traffic model
    double speedup_vs_baseline = 0.0;

    ClockSample clocks_before, clocks_after;
    std::string precision_label;
};

// Achievable ceiling: min(compute peak, bandwidth * ideal-traffic intensity).
inline double attn_roofline_gflops(const AttnProblem& p, Precision prec) {
    const int dtype_bytes = (prec == Precision::FP32) ? 4 : 2;
    const double compute_ceiling_gflops =
        (prec == Precision::FP32) ? gb10::kFp32PeakTFLOPS   * 1000.0
                                  : gb10::kBf16TensorTFLOPS * 1000.0;
    const double ai = attention_intensity_ideal(p.batch, p.heads, p.seq_len,
                                                p.head_dim, p.causal, dtype_bytes);
    const double mem_ceiling_gflops = gb10::kDramBandwidthMeasuredGBs * ai;
    return std::min(compute_ceiling_gflops, mem_ceiling_gflops);
}

class AttnBenchmarker {
public:
    AttnBenchmarker(const AttnProblem& prob, const BenchConfig& cfg)
        : prob_(prob), cfg_(cfg) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreate(&ev_start_));
        CUDA_CHECK(cudaEventCreate(&ev_stop_));
    }

    ~AttnBenchmarker() {
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
        cudaStreamDestroy(stream_);
    }

    // Computed once per problem and reused for every rung.
    void prepare_reference(const std::vector<float>& hQ,
                           const std::vector<float>& hK,
                           const std::vector<float>& hV) {
        reference_.assign(prob_.qkv_elems(), 0.0f);
        std::printf("  computing CPU reference (%s)... ", prob_.label().c_str());
        std::fflush(stdout);
        const auto t0 = std::chrono::steady_clock::now();
        attention_cpu(prob_.batch, prob_.heads, prob_.seq_len, prob_.head_dim,
                      prob_.scale(), prob_.causal,
                      hQ.data(), hK.data(), hV.data(), reference_.data());
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("%.1fs\n", std::chrono::duration<double>(t1 - t0).count());
        have_reference_ = true;
    }

    AttnBenchResult run(const AttnKernelSpec& spec, const AttnBuffers& buf) {
        AttnBenchResult r;
        r.kernel_name = spec.name;
        r.description = spec.description;
        r.problem     = prob_;
        r.precision_label = (spec.precision == Precision::FP32) ? "fp32" : "bf16";

        if (!spec.implemented) { r.skipped = true; return r; }

        AttnArgs args{prob_.batch, prob_.heads, prob_.seq_len, prob_.head_dim,
                      prob_.scale(), prob_.causal,
                      buf.Q, buf.K, buf.V,
                      buf.Q_bf16, buf.K_bf16, buf.V_bf16,
                      buf.O, buf.scratch, buf.scratch_elems};

        // Poison O with 0x7F bytes (~3.4e38 per float) so unwritten output cannot pass compare().
        if (cfg_.verify && have_reference_) {
            CUDA_CHECK(cudaMemsetAsync(buf.O, 0x7F, buf.bytes_qkv, stream_));
            spec.launch(args, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            CUDA_CHECK_KERNEL();

            std::vector<float> host_out(prob_.qkv_elems());
            CUDA_CHECK(cudaMemcpy(host_out.data(), buf.O, buf.bytes_qkv,
                                  cudaMemcpyDeviceToHost));
            r.correctness = compare(host_out, reference_, spec.tolerance);
            if (!r.correctness.passed) return r;      // never benchmark a wrong kernel
        } else {
            r.correctness.passed = true;
        }

        for (int i = 0; i < cfg_.warmup_iters; ++i) spec.launch(args, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK_KERNEL();

        r.clocks_before = sample_clocks();

        // O is fully overwritten each run; a rung that accumulates into O would need a restore here.
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

        const double flops = attention_flops(prob_.batch, prob_.heads,
                                             prob_.seq_len, prob_.head_dim,
                                             prob_.causal);
        r.gflops_p50 = gflops(flops, r.timing.p50_ms);
        r.gflops_min = gflops(flops, r.timing.min_ms);

        const int dtype_bytes = (spec.precision == Precision::FP32) ? 4 : 2;
        r.achieved_gbs = gbytes_per_s(
            attention_ideal_bytes(prob_.batch, prob_.heads, prob_.seq_len,
                                  prob_.head_dim, dtype_bytes),
            r.timing.p50_ms);

        const double ceiling = attn_roofline_gflops(prob_, spec.precision);
        r.pct_of_peak = (ceiling > 0.0) ? 100.0 * r.gflops_p50 / ceiling : 0.0;

        // First successfully benchmarked rung becomes the speedup baseline (so --only still works).
        if (baseline_gflops_ <= 0.0) baseline_gflops_ = r.gflops_p50;
        if (baseline_gflops_ > 0.0)  r.speedup_vs_baseline = r.gflops_p50 / baseline_gflops_;

        return r;
    }

    cudaStream_t stream() const { return stream_; }

private:
    AttnProblem        prob_;
    BenchConfig        cfg_;
    cudaStream_t       stream_{};
    cudaEvent_t        ev_start_{}, ev_stop_{};
    std::vector<float> reference_;
    bool               have_reference_  = false;
    double             baseline_gflops_ = 0.0;
};

inline void print_attn_header() {
    std::printf("\n%-24s %-6s %9s %9s %9s %8s %9s %8s %7s  %s\n",
                "kernel", "prec", "p50 (ms)", "min (ms)", "GFLOP/s",
                "%roof", "vs rung0", "eff GB/s", "cv", "status");
    std::printf("%s\n", std::string(122, '-').c_str());
}

inline void print_attn_result(const AttnBenchResult& r) {
    if (r.skipped) {
        std::printf("%-24s %-6s %9s %9s %9s %8s %9s %8s %7s  %s\n",
            r.kernel_name.c_str(), "-", "-", "-", "-", "-", "-", "-", "-",
            "not implemented (skipped)");
        return;
    }
    if (!r.correctness.passed) {
        std::printf("%-24s %-6s  WRONG: max rel err %.3g",
            r.kernel_name.c_str(), r.precision_label.c_str(),
            r.correctness.max_rel_error);
        if (r.correctness.first_bad_index >= 0) {
            const int D = r.problem.head_dim, S = r.problem.seq_len,
                      H = r.problem.heads;
            const long long idx = r.correctness.first_bad_index;
            const int d = static_cast<int>(idx % D);
            const int s = static_cast<int>((idx / D) % S);
            const int h = static_cast<int>((idx / D / S) % H);
            const int b = static_cast<int>(idx / D / S / H);
            std::printf("  (first bad at b=%d h=%d s=%d d=%d: got %.6g want %.6g)",
                        b, h, s, d,
                        r.correctness.first_bad_got, r.correctness.first_bad_want);
        }
        std::printf("\n");
        return;
    }
    const char* health = (r.timing.cv() > 0.05)
                       ? "  [NOISY: cv>5%, rerun on a quiet box]" : "";
    std::printf("%-24s %-6s %9.3f %9.3f %9.1f %7.1f%% %8.2fx %8.1f %7.3f  ok%s\n",
        r.kernel_name.c_str(), r.precision_label.c_str(),
        r.timing.p50_ms, r.timing.min_ms, r.gflops_p50,
        r.pct_of_peak, r.speedup_vs_baseline, r.achieved_gbs,
        r.timing.cv(), health);
}

// One JSON object per line; keep format in sync with the GEMM harness for bench/report.py.
inline void append_attn_jsonl(const std::string& path, const AttnBenchResult& r) {
    std::FILE* f = std::fopen(path.c_str(), "a");
    if (!f) {
        std::fprintf(stderr, "warning: could not open %s for append\n", path.c_str());
        return;
    }
    const std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    std::fprintf(f,
        "{\"timestamp\":\"%sZ\",\"phase\":\"attention\",\"kernel\":\"%s\","
        "\"description\":\"%s\",\"B\":%d,\"H\":%d,\"S\":%d,\"D\":%d,"
        "\"causal\":%s,\"precision\":\"%s\","
        "\"skipped\":%s,\"correct\":%s,\"max_rel_error\":%.6g,"
        "\"p50_ms\":%.6f,\"min_ms\":%.6f,\"p90_ms\":%.6f,\"p99_ms\":%.6f,"
        "\"mean_ms\":%.6f,\"stddev_ms\":%.6f,\"cv\":%.6f,\"n\":%d,"
        "\"gflops_p50\":%.3f,\"gflops_min\":%.3f,\"pct_of_roofline\":%.3f,"
        "\"effective_gbs\":%.3f,\"speedup_vs_rung0\":%.4f,"
        "\"sm_clock_before_mhz\":%.1f,\"sm_clock_after_mhz\":%.1f,"
        "\"power_w_after\":%.1f,\"temp_c_after\":%.1f}\n",
        ts, r.kernel_name.c_str(), r.description.c_str(),
        r.problem.batch, r.problem.heads, r.problem.seq_len, r.problem.head_dim,
        r.problem.causal ? "true" : "false", r.precision_label.c_str(),
        r.skipped ? "true" : "false",
        r.correctness.passed ? "true" : "false", r.correctness.max_rel_error,
        r.timing.p50_ms, r.timing.min_ms, r.timing.p90_ms, r.timing.p99_ms,
        r.timing.mean_ms, r.timing.stddev_ms, r.timing.cv(), r.timing.n,
        r.gflops_p50, r.gflops_min, r.pct_of_peak, r.achieved_gbs,
        r.speedup_vs_baseline,
        r.clocks_before.sm_clock_mhz, r.clocks_after.sm_clock_mhz,
        r.clocks_after.power_w, r.clocks_after.temp_c);
    std::fclose(f);
}

}  // namespace ladder
