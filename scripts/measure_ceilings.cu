// measure_ceilings.cu: measure DRAM read bandwidth and fp32 FMA ceilings.
// Keep in sync with src/common/gb10.hpp, bench/report.py, bench/sweep.py.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x)                                                              \
    do {                                                                      \
        cudaError_t err_ = (x);                                               \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                 \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace {

// Grid-stride float4 reads; sink write defeats dead-code elimination.
__global__ void read_bw_kernel(const float4* __restrict__ buf,
                               size_t n4, float* __restrict__ sink) {
    float acc = 0.0f;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n4; i += stride) {
        const float4 v = buf[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -1.0f) *sink = acc;   // data-dependent, never taken
}

// 8 independent FMA chains per thread avoid a serial dependence bottleneck.
__global__ void fma_kernel(float seed, int iters, float* __restrict__ sink) {
    float a0 = seed, a1 = seed + 1, a2 = seed + 2, a3 = seed + 3;
    float a4 = seed + 4, a5 = seed + 5, a6 = seed + 6, a7 = seed + 7;
    const float b = 1.000001f, c = 1e-7f;
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c);
        a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
        a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c);
        a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
    }
    const float r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (r == -1.0f) *sink = r;
}

float time_ms_best_of(int reps, void (*launch)(void*), void* ctx) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    float best = 1e30f;
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaEventRecord(t0));
        launch(ctx);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return best;
}

struct BwCtx { const float4* buf; size_t n4; float* sink; };
struct FmaCtx { int iters; float* sink; dim3 grid, block; };

void launch_bw(void* p) {
    auto* c = static_cast<BwCtx*>(p);
    read_bw_kernel<<<48 * 8, 256>>>(c->buf, c->n4, c->sink);
}

void launch_fma(void* p) {
    auto* c = static_cast<FmaCtx*>(p);
    fma_kernel<<<c->grid, c->block>>>(1.0f, c->iters, c->sink);
}

}  // namespace

int main() {
    const size_t bytes = 8ull << 30;   // 8 GiB: 340x the 24 MB L2
    const size_t n4 = bytes / sizeof(float4);

    float4* buf = nullptr;
    float* sink = nullptr;
    CHECK(cudaMalloc(&buf, bytes));
    CHECK(cudaMalloc(&sink, sizeof(float)));
    CHECK(cudaMemset(buf, 1, bytes));

    BwCtx bw{buf, n4, sink};
    launch_bw(&bw);                        // warm up
    CHECK(cudaDeviceSynchronize());
    const float bw_ms = time_ms_best_of(5, launch_bw, &bw);
    const double gbs = bytes / (bw_ms * 1e-3) / 1e9;

    FmaCtx fm{20000, sink, dim3(48 * 16), dim3(256)};
    launch_fma(&fm);                       // warm up
    CHECK(cudaDeviceSynchronize());
    const float fma_ms = time_ms_best_of(5, launch_fma, &fm);
    const double threads = 48.0 * 16 * 256;
    const double flops = threads * fm.iters * 8.0 * 2.0;
    const double tflops = flops / (fma_ms * 1e-3) / 1e12;

    std::printf("== measured ceilings (best of 5) ==\n");
    std::printf("  DRAM read bandwidth : %7.1f GB/s   (gb10.hpp says 231, spec 273)\n", gbs);
    std::printf("  fp32 FMA throughput : %7.2f TFLOP/s (gb10.hpp says 29.7, derived)\n", tflops);
    std::printf("\n(bf16 tensor peak 212.9 TF was measured with mmapeak; rerun that\n");
    std::printf(" separately if you doubt it.)\n");
    std::printf("If these disagree with gb10.hpp, update src/common/gb10.hpp,\n");
    std::printf("bench/report.py and bench/sweep.py TOGETHER.\n");

    CHECK(cudaFree(buf));
    CHECK(cudaFree(sink));
    return 0;
}
