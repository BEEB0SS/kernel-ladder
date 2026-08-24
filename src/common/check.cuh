// check.cuh: CUDA/cuBLAS error-checking macros.

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                "\n[CUDA ERROR] %s\n  %s:%d\n  in: %s\n",                      \
                cudaGetErrorString(_err), __FILE__, __LINE__, #expr);          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Checks launch and execution errors; synchronizes the device, so keep out of timing loops.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t _launch = cudaGetLastError();                              \
        if (_launch != cudaSuccess) {                                          \
            std::fprintf(stderr,                                               \
                "\n[KERNEL LAUNCH FAILED] %s\n  %s:%d\n"                       \
                "  This is a LAUNCH error, not an execution error. Usual causes:\n" \
                "    - grid or block dimension is 0 or exceeds the limit (1024 threads/block)\n" \
                "    - dynamic shared memory request exceeds 99KB (the GB10 per-block cap)\n" \
                "    - the kernel used too many registers for the requested block size\n", \
                cudaGetErrorString(_launch), __FILE__, __LINE__);              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
        cudaError_t _exec = cudaDeviceSynchronize();                           \
        if (_exec != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "\n[KERNEL EXECUTION FAILED] %s\n  %s:%d\n"                    \
                "  This is a RUNTIME error inside the kernel. Usual causes:\n" \
                "    - out-of-bounds global or shared memory access\n"         \
                "    - missing bounds guard on the M/N/K edges when the matrix\n" \
                "      size is not an exact multiple of the tile size\n"      \
                "  Run under `compute-sanitizer ./build/ladder ...` to get the\n" \
                "  exact offending line. It is worth the 30x slowdown.\n",     \
                cudaGetErrorString(_exec), __FILE__, __LINE__);                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

inline const char* cublasErrorString(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "unknown cublas status";
    }
}

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "\n[cuBLAS ERROR] %s\n  %s:%d\n  in: %s\n",   \
                cublasErrorString(_s), __FILE__, __LINE__, #expr);             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
