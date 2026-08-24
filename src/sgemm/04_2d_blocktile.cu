// Rung 4: 2D blocktile — each thread computes a TM x TN register patch of C.

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

// #ifndef guards let scripts/sweep_tiles.sh override the tile shape via -D flags.
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16   // GB10 sweep winner (8 -> 16)
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif

constexpr int kThreads = (BM * BN) / (TM * TN);

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");
static_assert((BM * BK) % kThreads == 0, "A-tile must divide evenly among threads");
static_assert((BK * BN) % kThreads == 0, "B-tile must divide evenly among threads");
static_assert((BM * BK + BK * BN) * 4 <= 99 * 1024,
              "shared memory exceeds the GB10 per-block cap of 99KB");
static_assert(kThreads <= 1024,
              "(BM*BN)/(TM*TN) exceeds the 1024 threads-per-block limit");
static_assert(kThreads > 0 && kThreads % 32 == 0,
              "thread count must be a positive multiple of the 32-wide warp");

__global__ __launch_bounds__(kThreads)
void blocktile_2d_sgemm_kernel(int M, int N, int K,
                               float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float beta,
                               float* __restrict__ C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float acc[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];

    // Each thread owns a TM x TN patch of the block tile.
    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Out-of-range lanes store 0.0f so every thread reaches the barriers.
        for (int idx = threadIdx.x; idx < BM * BK; idx += kThreads) {
            const int r = idx / BK, c = idx % BK;
            const int gr = blockIdx.y * BM + r, gc = k0 + c;
            As[idx] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += kThreads) {
            const int r = idx / BN, c = idx % BN;
            const int gr = k0 + r, gc = blockIdx.x * BN + c;
            Bs[idx] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // The unrolls keep acc/regA/regB in named registers, not local mem.
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[(thread_row * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[k * BN + thread_col * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = blockIdx.y * BM + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = blockIdx.x * BN + thread_col * TN + j;
            if (r < M && c < N)
                C[r * N + c] = alpha * acc[i][j] + beta * C[r * N + c];
        }
    }
}

}  // namespace

void launch_2d_blocktile(const GemmArgs& a, cudaStream_t s) {
    dim3 block(kThreads);
    dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));
    blocktile_2d_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);
}

}  // namespace ladder
