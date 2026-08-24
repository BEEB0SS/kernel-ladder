// CPU reference oracle for scaled dot-product attention, plus FLOP/byte accounting.

#pragma once

#include "../common/cpu_reference.hpp"   // make_matrix, compare, CompareResult

#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace ladder {

// Q, K, V, O are [B][H][S][D] row-major (PyTorch post-transpose layout); scores are [B][H][S][S].
inline std::size_t qkv_index(int b, int h, int s, int d,
                             int H, int S, int D) {
    return ((static_cast<std::size_t>(b) * H + h) * S + s) * D + d;
}

inline std::size_t score_index(int b, int h, int i, int j,
                               int H, int S) {
    return ((static_cast<std::size_t>(b) * H + h) * S + i) * S + j;
}

// O = softmax(Q @ K^T * scale [+ causal mask]) @ V.
// `scale` is passed in so oracle, harness and kernels provably share one constant.
// Intermediates are double: the oracle must out-precision every kernel it judges.
inline void attention_cpu(int B, int H, int S, int D,
                          float scale,
                          bool causal,
                          const float* Q,
                          const float* K,
                          const float* V,
                          float* O) {
    std::vector<double> row(static_cast<std::size_t>(S));

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < S; ++i) {

                const int j_end = causal ? (i + 1) : S;

                double m = -std::numeric_limits<double>::infinity();
                for (int j = 0; j < j_end; ++j) {
                    double acc = 0.0;
                    for (int d = 0; d < D; ++d) {
                        acc += static_cast<double>(Q[qkv_index(b, h, i, d, H, S, D)]) *
                               static_cast<double>(K[qkv_index(b, h, j, d, H, S, D)]);
                    }
                    acc *= static_cast<double>(scale);
                    row[j] = acc;
                    if (acc > m) m = acc;
                }

                // Subtracting the row max keeps exp in (0,1], so l >= 1 and no epsilon guard is needed.
                double l = 0.0;
                for (int j = 0; j < j_end; ++j) {
                    row[j] = std::exp(row[j] - m);
                    l += row[j];
                }
                const double inv_l = 1.0 / l;

                for (int d = 0; d < D; ++d) {
                    double acc = 0.0;
                    for (int j = 0; j < j_end; ++j) {
                        acc += row[j] * inv_l *
                               static_cast<double>(V[qkv_index(b, h, j, d, H, S, D)]);
                    }
                    O[qkv_index(b, h, i, d, H, S, D)] = static_cast<float>(acc);
                }

            }
        }
    }
}

// Reference post-softmax probabilities P, masked entries 0 (for diffing intermediates).
inline std::vector<float> attention_probs_cpu(int B, int H, int S, int D,
                                              float scale, bool causal,
                                              const float* Q, const float* K) {
    std::vector<float> P(static_cast<std::size_t>(B) * H * S * S, 0.0f);
    std::vector<double> row(static_cast<std::size_t>(S));

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int i = 0; i < S; ++i) {
        const int j_end = causal ? (i + 1) : S;
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < j_end; ++j) {
            double acc = 0.0;
            for (int d = 0; d < D; ++d)
                acc += static_cast<double>(Q[qkv_index(b, h, i, d, H, S, D)]) *
                       static_cast<double>(K[qkv_index(b, h, j, d, H, S, D)]);
            acc *= static_cast<double>(scale);
            row[j] = acc;
            if (acc > m) m = acc;
        }
        double l = 0.0;
        for (int j = 0; j < j_end; ++j) { row[j] = std::exp(row[j] - m); l += row[j]; }
        for (int j = 0; j < j_end; ++j)
            P[score_index(b, h, i, j, H, S)] = static_cast<float>(row[j] / l);
    }
    return P;
}

// FLOPs = 4*B*H*S*S*D (softmax excluded by convention); causal uses exact S(S+1)/2 pairs.
// Causal count assumes the kernel skips masked tiles; a mask-then-compute kernel is flattered ~2x.
inline double attention_flops(int B, int H, int S, int D, bool causal) {
    const double bh = static_cast<double>(B) * static_cast<double>(H);
    const double pairs = causal
        ? (static_cast<double>(S) * (static_cast<double>(S) + 1.0) * 0.5)
        : (static_cast<double>(S) * static_cast<double>(S));
    return 4.0 * bh * pairs * static_cast<double>(D);
}

// Ideal fused traffic: Q, K, V read once, O written once.
inline double attention_ideal_bytes(int B, int H, int S, int D,
                                    int dtype_bytes = 4) {
    return 4.0 * static_cast<double>(B) * H * S * D * dtype_bytes;
}

// Naive traffic: the S x S scores/probs round-trip DRAM; assumes zero cache reuse across passes.
inline double attention_naive_bytes(int B, int H, int S, int D,
                                    int dtype_bytes = 4) {
    const double bh = static_cast<double>(B) * static_cast<double>(H);
    return (4.0 * static_cast<double>(S) * D +
            4.0 * static_cast<double>(S) * S) * bh * dtype_bytes;
}

inline double score_matrix_bytes(int B, int H, int S, int dtype_bytes = 4) {
    return static_cast<double>(B) * H * S * S * dtype_bytes;
}

inline double attention_intensity_ideal(int B, int H, int S, int D,
                                        bool causal, int dtype_bytes = 4) {
    return attention_flops(B, H, S, D, causal) /
           attention_ideal_bytes(B, H, S, D, dtype_bytes);
}

inline double attention_intensity_naive(int B, int H, int S, int D,
                                        bool causal, int dtype_bytes = 4) {
    return attention_flops(B, H, S, D, causal) /
           attention_naive_bytes(B, H, S, D, dtype_bytes);
}

// Kept as a function so oracle, harness and every kernel share one definition of the scale.
inline float attention_scale(int D) {
    return 1.0f / std::sqrt(static_cast<float>(D));
}

inline std::vector<float> make_qkv(int B, int H, int S, int D, unsigned seed) {
    return make_matrix(B * H * S, D, seed);
}

}  // namespace ladder
