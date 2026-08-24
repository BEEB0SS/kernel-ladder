// Rung 0: naive three-kernel attention (Q@K^T, row softmax, P@V) with a B*H*S*S scratch in DRAM.

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cfloat>
#include <cmath>

namespace ladder {
namespace {

// All kernels: blockIdx.z carries the flattened (b*H + h), so H never appears in the arithmetic.
constexpr int kScoreThreads = 256;

__global__ void qk_scores_kernel(int S, int D,
                                 float scale, bool causal,
                                 const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 float* __restrict__ scores) {
    const int bh = blockIdx.z;
    const int i  = blockIdx.y;
    const int j  = blockIdx.x * kScoreThreads + threadIdx.x;
    if (j >= S) return;

    const long long qkv_base = static_cast<long long>(bh) * S * D;
    const long long sc_base  = static_cast<long long>(bh) * S * S;

    // Early return (not mask-after-compute) keeps FLOPs matching attention_flops(causal=true).
    if (causal && j > i) {
        // -INFINITY so exp(-inf - m) is exactly 0; causal rows always have key i visible, so never all-masked.
        scores[sc_base + static_cast<long long>(i) * S + j] = -INFINITY;
        return;
    }

    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        acc += Q[qkv_base + static_cast<long long>(i) * D + d] *
               K[qkv_base + static_cast<long long>(j) * D + d];
    }
    scores[sc_base + static_cast<long long>(i) * S + j] = acc * scale;
}

// In-place row softmax, one thread per row: max, exp-and-sum, normalize.
__global__ void softmax_rows_kernel(int S, long long total_rows,
                                    float* __restrict__ scores) {
    const long long r = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (r >= total_rows) return;  // r indexes (b,h,i) flattened

    float* row = scores + r * S;

    float m = -FLT_MAX;
    for (int j = 0; j < S; ++j) m = fmaxf(m, row[j]);

    float l = 0.0f;
    for (int j = 0; j < S; ++j) {
        const float e = expf(row[j] - m);  // shift by the row max so exp cannot overflow
        row[j] = e;
        l += e;
    }

    // l >= 1 (the max element contributes exp(0)=1), so no epsilon guard is needed.
    const float inv_l = 1.0f / l;
    for (int j = 0; j < S; ++j) row[j] *= inv_l;
}

__global__ void pv_kernel(int S, int D,
                          const float* __restrict__ P,
                          const float* __restrict__ V,
                          float* __restrict__ O) {
    const int bh = blockIdx.z;
    const int i  = blockIdx.y;
    const int d  = threadIdx.x;
    if (d >= D) return;

    const long long qkv_base = static_cast<long long>(bh) * S * D;
    const long long sc_base  = static_cast<long long>(bh) * S * S;

    float acc = 0.0f;
    for (int j = 0; j < S; ++j) {
        // Masked entries are exactly 0 after softmax, so the causal case needs no branch here.
        acc += P[sc_base + static_cast<long long>(i) * S + j] *
               V[qkv_base + static_cast<long long>(j) * D + d];
    }
    O[qkv_base + static_cast<long long>(i) * D + d] = acc;
}

}  // namespace

void launch_attn_naive(const AttnArgs& a, cudaStream_t s) {
    if (a.scratch == nullptr ||
        a.scratch_elems < static_cast<std::size_t>(a.B) * a.H * a.S * a.S) {
        std::fprintf(stderr,
            "00_naive_attention: needs a B*H*S*S scratch buffer (%.2f GB at this "
            "size) and did not get one.\n",
            static_cast<double>(a.B) * a.H * a.S * a.S * 4.0 / 1e9);
        std::exit(EXIT_FAILURE);
    }

    const int  BH = a.B * a.H;
    const long long rows = static_cast<long long>(BH) * a.S;

    {
        dim3 block(kScoreThreads);
        dim3 grid(ceil_div(a.S, kScoreThreads), a.S, BH);
        qk_scores_kernel<<<grid, block, 0, s>>>(
            a.S, a.D, a.scale, a.causal, a.Q, a.K, a.scratch);
    }

    {
        const int threads = 256;
        const int blocks  = static_cast<int>((rows + threads - 1) / threads);
        softmax_rows_kernel<<<blocks, threads, 0, s>>>(a.S, rows, a.scratch);
    }

    {
        const int threads = ((a.D + 31) / 32) * 32;  // round D up to a full warp
        dim3 block(threads);
        dim3 grid(1, a.S, BH);
        pv_kernel<<<grid, block, 0, s>>>(a.S, a.D, a.scratch, a.V, a.O);
    }

    // No CUDA_CHECK_KERNEL here: its cudaDeviceSynchronize would wreck the timing loop.
}

}  // namespace ladder
