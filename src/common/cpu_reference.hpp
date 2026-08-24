// CPU reference oracle: deliberately simple and slow, accumulates in double.

#pragma once

#include <cstddef>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

namespace ladder {

// C = alpha * A @ B + beta * C; row-major: A[m*K+k], B[k*N+n], C[m*N+n].
inline void sgemm_cpu(int M, int N, int K,
                      float alpha,
                      const float* A,
                      const float* B,
                      float beta,
                      float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;                       // double on purpose: oracle must out-precision what it judges
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(A[m * K + k]) *
                       static_cast<double>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(alpha * acc +
                                              beta * static_cast<double>(C[m * N + n]));
        }
    }
}

struct CompareResult {
    bool   passed          = false;
    double max_rel_error   = 0.0;
    double mean_rel_error  = 0.0;
    int    num_mismatches  = 0;
    int    first_bad_index = -1;
    float  first_bad_got   = 0.0f;
    float  first_bad_want  = 0.0f;
};

// `scale` floors the rel-error denominator at the problem's natural magnitude (sqrt(K) for GEMM); a bare relative error fails even cuBLAS on near-zero outputs.
inline CompareResult compare(const std::vector<float>& got,
                             const std::vector<float>& want,
                             double tolerance,
                             double scale = 1.0) {
    CompareResult r;
    if (got.size() != want.size()) return r;   // passed stays false
    if (scale < 1e-30) scale = 1e-30;

    double sum_rel = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::isnan(got[i]) || std::isinf(got[i])) {
            ++r.num_mismatches;
            if (r.first_bad_index < 0) {
                r.first_bad_index = static_cast<int>(i);
                r.first_bad_got   = got[i];
                r.first_bad_want  = want[i];
            }
            r.max_rel_error = std::numeric_limits<double>::infinity();
            continue;
        }
        const double diff = std::fabs(static_cast<double>(got[i]) -
                                      static_cast<double>(want[i]));
        const double rel  = diff / std::max(std::fabs(static_cast<double>(want[i])), scale);
        sum_rel += rel;
        if (rel > r.max_rel_error) r.max_rel_error = rel;
        if (rel > tolerance) {
            ++r.num_mismatches;
            if (r.first_bad_index < 0) {
                r.first_bad_index = static_cast<int>(i);
                r.first_bad_got   = got[i];
                r.first_bad_want  = want[i];
            }
        }
    }
    r.mean_rel_error = got.empty() ? 0.0 : sum_rel / static_cast<double>(got.size());
    r.passed = (r.num_mismatches == 0);
    return r;
}

// Fixed seed for reproducibility; normal dist gives mixed-sign data that exposes sign/init bugs.
inline std::vector<float> make_matrix(int rows, int cols, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> m(static_cast<std::size_t>(rows) * cols);
    for (auto& v : m) v = dist(gen);
    return m;
}

// 2*M*N*K; alpha/beta scaling ignored to match published GEMM benchmarks.
inline double gemm_flops(int M, int N, int K) {
    return 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
}

// Ideal DRAM traffic assuming perfect caching: every element moved exactly once.
inline double gemm_min_bytes(int M, int N, int K, int dtype_bytes = 4) {
    return (static_cast<double>(M) * K +
            static_cast<double>(K) * N +
            static_cast<double>(M) * N) * dtype_bytes;
}

inline double arithmetic_intensity(int M, int N, int K, int dtype_bytes = 4) {
    return gemm_flops(M, N, K) / gemm_min_bytes(M, N, K, dtype_bytes);
}

}  // namespace ladder
