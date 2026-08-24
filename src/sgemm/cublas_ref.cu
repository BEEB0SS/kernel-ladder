// cuBLAS baseline: strict-fp32 and TF32 reference GEMMs.

#include "kernels.cuh"
#include "../common/check.cuh"
#include <cublas_v2.h>

namespace ladder {
namespace {
cublasHandle_t g_handle = nullptr;
}  // namespace

void cublas_init() {
    if (!g_handle) CUBLAS_CHECK(cublasCreate(&g_handle));
}

void cublas_destroy() {
    if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }
}

void launch_cublas(const GemmArgs& a, cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_handle, s));
    // PEDANTIC_MATH forbids silent fp32->TF32 demotion; the default varies by CUDA version.
    CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_PEDANTIC_MATH));

    // Row-major C = A@B via column-major C^T = B^T A^T: swap M/N and pass B first.
    CUBLAS_CHECK(cublasSgemm(
        g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        a.N, a.M, a.K,
        &a.alpha,
        a.B_f32, a.N,
        a.A_f32, a.K,
        &a.beta,
        a.C, a.N));
}

void launch_cublas_tf32(const GemmArgs& a, cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(g_handle, s));
    // TF32 baseline: the fair comparison for reduced-precision kernels.
    CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    CUBLAS_CHECK(cublasSgemm(
        g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        a.N, a.M, a.K,
        &a.alpha,
        a.B_f32, a.N,
        a.A_f32, a.K,
        &a.beta,
        a.C, a.N));
}

}  // namespace ladder
