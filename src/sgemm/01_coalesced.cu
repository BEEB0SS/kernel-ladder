// Rung 1: coalesced global loads via threadIdx.x -> column mapping.

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

constexpr int kBlockSize = 32;

__global__ void coalesced_sgemm_kernel(int M, int N, int K,
                                       float alpha,
                                       const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float beta,
                                       float* __restrict__ C) {
    // threadIdx.x must vary along the contiguous (column) dimension: B reads coalesce, A reads broadcast.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

}  // namespace

void launch_coalesced(const GemmArgs& a, cudaStream_t s) {
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid(ceil_div(a.N, kBlockSize),
              ceil_div(a.M, kBlockSize));

    coalesced_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);
}

}  // namespace ladder
