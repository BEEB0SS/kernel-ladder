// Rung 0: naive SGEMM, one thread per output element.

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

constexpr int kBlockSize = 32;

__global__ void naive_sgemm_kernel(int M, int N, int K,
                                   float alpha,
                                   const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float beta,
                                   float* __restrict__ C) {
    // x -> row is deliberate: consecutive warp threads stride A by K (uncoalesced).
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid is rounded up, so edge blocks have out-of-range threads.
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }

    // Reads C even when beta == 0; the harness always initializes C.
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

}  // namespace

void launch_naive(const GemmArgs& a, cudaStream_t s) {
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid(ceil_div(a.M, kBlockSize),
              ceil_div(a.N, kBlockSize));

    naive_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);

    // No CUDA_CHECK_KERNEL here: its sync would serialize the timing loop;
    // the harness error-checks one launch during the correctness pass.
}

}  // namespace ladder
