// Rung 2: shared-memory tiled SGEMM (BM x BK and BK x BN tiles staged in smem).

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

// sweep_tiles.sh passes -DBM/-DBN/-DBK to every TU; only rung 4 consumes them.
#ifdef BM
#undef BM
#endif
#ifdef BN
#undef BN
#endif
#ifdef BK
#undef BK
#endif

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

__global__ void smem_tiled_sgemm_kernel(int M, int N, int K,
                                        float alpha,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float beta,
                                        float* __restrict__ C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * BN + tx;
    const int row = blockIdx.y * BM + ty;

    // No early return: every thread must reach every __syncthreads() below.
    const bool in_range = (row < M && col < N);

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Out-of-range lanes pad with 0.0f, which contributes nothing to the sum.
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + (k0 + tx)] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();   // all loads complete

        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // all reads complete before the tiles are overwritten
    }

    if (in_range) {
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

}  // namespace

void launch_smem_tiled(const GemmArgs& a, cudaStream_t s) {
    dim3 block(BN, BM);
    dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));
    smem_tiled_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);
}

}  // namespace ladder
