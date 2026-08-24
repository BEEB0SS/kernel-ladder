// 03_1d_blocktile.cu — Rung 3: 1D blocktile, each thread computes TM outputs.

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

// sweep_tiles.sh passes -DBM=... to every TU; only rung 4 consumes them, so drop here.
#ifdef BM
#undef BM
#endif
#ifdef BN
#undef BN
#endif
#ifdef BK
#undef BK
#endif
#ifdef TM
#undef TM
#endif

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;    // results per thread, down the M dimension

constexpr int kThreads = (BM * BN) / TM;   // each thread owns TM rows of one column

__global__ __launch_bounds__(kThreads)
void blocktile_1d_sgemm_kernel(int M, int N, int K,
                               float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float beta,
                               float* __restrict__ C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float acc[TM] = {0.0f};

    // Compute mapping: TM consecutive rows of one column per thread.
    const int thread_col = threadIdx.x % BN;
    const int thread_row = threadIdx.x / BN;

    // Load mapping is decoupled from the compute mapping.
    const int a_load_row = threadIdx.x / BK;
    const int a_load_col = threadIdx.x % BK;
    const int b_load_row = threadIdx.x / BN;
    const int b_load_col = threadIdx.x % BN;

    const int col  = blockIdx.x * BN + thread_col;
    const int row0 = blockIdx.y * BM + thread_row * TM;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Out-of-range lanes store 0.0f so every thread reaches the barriers.
        const int ga_row = blockIdx.y * BM + a_load_row;
        const int ga_col = k0 + a_load_col;
        As[a_load_row * BK + a_load_col] =
            (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.0f;

        const int gb_row = k0 + b_load_row;
        const int gb_col = blockIdx.x * BN + b_load_col;
        Bs[b_load_row * BN + b_load_col] =
            (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            const float b = Bs[k * BN + thread_col];   // one load feeds TM FMAs
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                acc[i] += As[(thread_row * TM + i) * BK + k] * b;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + i;
        if (r < M && col < N)
            C[r * N + col] = alpha * acc[i] + beta * C[r * N + col];
    }
}

}  // namespace

void launch_1d_blocktile(const GemmArgs& a, cudaStream_t s) {
    dim3 block(kThreads);
    dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));
    blocktile_1d_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);
}

}  // namespace ladder
