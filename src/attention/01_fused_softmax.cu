// Rung 1: fused QK^T + softmax with warp-shuffle row reductions.

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cfloat>
#include <cmath>

namespace ladder {
namespace {

constexpr int kRowThreads = 256;
constexpr int kWarpsPerBlock = kRowThreads / 32;

// After a warp reduction, lane 0 holds the result; the other lanes hold garbage.
__device__ __forceinline__ float warp_reduce_max(float v) {
    // All 32 lanes must reach this shuffle -- no divergent early return above it.
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, delta));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, delta);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float smem[kWarpsPerBlock];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    // -FLT_MAX identity: 0.0f would break the instant a row is all-negative.
    v = (threadIdx.x < kWarpsPerBlock) ? smem[threadIdx.x] : -FLT_MAX;
    if (warp == 0) v = warp_reduce_max(v);
    return v;   // valid in thread 0 only
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[kWarpsPerBlock];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();

    v = (threadIdx.x < kWarpsPerBlock) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;   // valid in thread 0 only
}

// Fused scores -> softmax, one block per query row; dynamic smem is O(S).
__global__ __launch_bounds__(kRowThreads)
void qk_softmax_kernel(int S, int D,
                       float scale, bool causal,
                       const float* __restrict__ Q,
                       const float* __restrict__ K,
                       float* __restrict__ probs) {
    extern __shared__ float smem[];
    float* q_row  = smem;          // [0 .. D-1]
    float* scores = smem + D;      // [D .. D+S-1]

    const int bh = blockIdx.y;
    const int i  = blockIdx.x;
    const long long qkv_base = static_cast<long long>(bh) * S * D;
    const long long row_base = (static_cast<long long>(bh) * S + i) * S;

    for (int d = threadIdx.x; d < D; d += kRowThreads)
        q_row[d] = Q[qkv_base + static_cast<long long>(i) * D + d];
    __syncthreads();

    // Masked lanes contribute the identity -- every thread must reach every shuffle/barrier.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < S; j += kRowThreads) {
        float sj = -FLT_MAX;   // finite sentinel: exp underflows to exactly 0
        if (!(causal && j > i)) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d)
                acc += q_row[d] * K[qkv_base + static_cast<long long>(j) * D + d];
            sj = acc * scale;
        }
        scores[j] = sj;
        local_max = fmaxf(local_max, sj);
    }

    // Block reductions leave the answer in thread 0 only; broadcast it.
    __shared__ float bcast;
    const float m0 = block_reduce_max(local_max);
    if (threadIdx.x == 0) bcast = m0;
    __syncthreads();
    const float m = bcast;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < S; j += kRowThreads) {
        const float e = (causal && j > i) ? 0.0f : expf(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }

    // block_reduce_sum's internal barrier also protects the bcast reuse.
    const float l0 = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) bcast = l0;
    __syncthreads();
    const float inv_l = 1.0f / bcast;

    for (int j = threadIdx.x; j < S; j += kRowThreads)
        probs[row_base + j] = scores[j] * inv_l;   // masked entries: exactly 0
}

// O = P @ V; identical to rung 0's kernel, kept self-contained.
__global__ void pv_kernel_r1(int S, int D,
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
        // Masked entries are exactly 0.0f after the softmax, so no branch.
        acc += P[sc_base + static_cast<long long>(i) * S + j] *
               V[qkv_base + static_cast<long long>(j) * D + d];
    }
    O[qkv_base + static_cast<long long>(i) * D + d] = acc;
}

}  // namespace

void launch_attn_fused_softmax(const AttnArgs& a, cudaStream_t s) {
    if (a.scratch == nullptr ||
        a.scratch_elems < static_cast<std::size_t>(a.B) * a.H * a.S * a.S) {
        std::fprintf(stderr,
            "01_fused_softmax: needs a B*H*S*S scratch buffer (%.2f GB at this "
            "size) and did not get one.\n",
            static_cast<double>(a.B) * a.H * a.S * a.S * 4.0 / 1e9);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t smem_bytes = (static_cast<std::size_t>(a.D) + a.S) * sizeof(float);
    if (smem_bytes > gb10::kMaxSmemPerBlockBytes) {
        std::fprintf(stderr,
            "01_fused_softmax: Q row + score row need %zu B of shared memory, "
            "over the %d B per-block cap. This rung's smem is O(S) by design — "
            "rung 2 removes that.\n",
            smem_bytes, gb10::kMaxSmemPerBlockBytes);
        std::exit(EXIT_FAILURE);
    }
    if (smem_bytes > 48 * 1024) {
        // Above 48 KB dynamic smem needs an explicit opt-in; configure once, not per iteration.
        static std::size_t configured = 0;
        if (configured != smem_bytes) {
            CUDA_CHECK(cudaFuncSetAttribute(qk_softmax_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            configured = smem_bytes;
        }
    }

    dim3 grid(a.S, a.B * a.H);
    qk_softmax_kernel<<<grid, kRowThreads, smem_bytes, s>>>(
        a.S, a.D, a.scale, a.causal, a.Q, a.K, a.scratch);

    const int threads = ((a.D + 31) / 32) * 32;
    dim3 pv_grid(1, a.S, a.B * a.H);
    pv_kernel_r1<<<pv_grid, threads, 0, s>>>(a.S, a.D, a.scratch, a.V, a.O);
}

}  // namespace ladder
