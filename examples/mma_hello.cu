// mma_hello.cu — one warp, one mma.sync.m16n8k16 (bf16 -> fp32), checked against a CPU loop.
// Build: nvcc -arch=sm_121 -O3 -o mma_hello docs/examples/mma_hello.cu

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(expr)                                                            \
    do {                                                                       \
        cudaError_t e = (expr);                                                \
        if (e != cudaSuccess) {                                                \
            std::printf("CUDA error %s at %s:%d\n",                            \
                        cudaGetErrorString(e), __FILE__, __LINE__);            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

constexpr int M = 16, N = 8, K = 16;

// Pack two bf16 into one 32-bit register; `lo` lands in the low 16 bits.
__device__ __forceinline__ unsigned pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return  static_cast<unsigned>(__bfloat16_as_ushort(lo))
         | (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16);
}

// A: 16x16 row-major [m][k]; B: 16x8 stored [k][n] (column-major operand for .row.col); D: 16x8 row-major.
__global__ void mma_hello_kernel(const __nv_bfloat16* __restrict__ A,
                                 const __nv_bfloat16* __restrict__ B,
                                 float* __restrict__ D) {
    const int lane    = threadIdx.x;
    const int groupID = lane / 4;
    const int tig     = lane % 4;

    unsigned a[4];
    a[0] = pack(A[(groupID    ) * K + 2 * tig    ], A[(groupID    ) * K + 2 * tig + 1]);
    a[1] = pack(A[(groupID + 8) * K + 2 * tig    ], A[(groupID + 8) * K + 2 * tig + 1]);
    a[2] = pack(A[(groupID    ) * K + 2 * tig + 8], A[(groupID    ) * K + 2 * tig + 9]);
    a[3] = pack(A[(groupID + 8) * K + 2 * tig + 8], A[(groupID + 8) * K + 2 * tig + 9]);

    // B fragment is indexed [k][n]: groupID selects the COLUMN (.row.col).
    unsigned b[2];
    b[0] = pack(B[(2 * tig    ) * N + groupID], B[(2 * tig + 1) * N + groupID]);
    b[1] = pack(B[(2 * tig + 8) * N + groupID], B[(2 * tig + 9) * N + groupID]);

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float d[4];

    // .aligned: all 32 lanes must execute this together; divergence here is UB.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "          // D
        "{%4, %5, %6, %7}, "          // A
        "{%8, %9}, "                  // B
        "{%10, %11, %12, %13};\n"     // C
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    D[(groupID    ) * N + 2 * tig    ] = d[0];
    D[(groupID    ) * N + 2 * tig + 1] = d[1];
    D[(groupID + 8) * N + 2 * tig    ] = d[2];
    D[(groupID + 8) * N + 2 * tig + 1] = d[3];
}

int main() {
    // Small-integer inputs are exact in bf16, so any mismatch is a layout bug, not rounding.
    float hA[M * K], hB[K * N], hRef[M * N];
    __nv_bfloat16 bA[M * K], bB[K * N];

    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            hA[m * K + k] = static_cast<float>((m + 2 * k) % 7 - 3);
            bA[m * K + k] = __float2bfloat16(hA[m * K + k]);
        }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            hB[k * N + n] = static_cast<float>((3 * k + n) % 5 - 2);
            bB[k * N + n] = __float2bfloat16(hB[k * N + n]);
        }

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += hA[m * K + k] * hB[k * N + n];
            hRef[m * N + n] = static_cast<float>(acc);
        }

    __nv_bfloat16 *dA, *dB;
    float* dD;
    CHECK(cudaMalloc(&dA, sizeof(bA)));
    CHECK(cudaMalloc(&dB, sizeof(bB)));
    CHECK(cudaMalloc(&dD, sizeof(hRef)));
    CHECK(cudaMemcpy(dA, bA, sizeof(bA), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, bB, sizeof(bB), cudaMemcpyHostToDevice));

    // Exactly one warp: the .aligned mma requires all 32 lanes.
    mma_hello_kernel<<<1, 32>>>(dA, dB, dD);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float hD[M * N];
    CHECK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

    std::printf("D = A(16x16) @ B(16x8), one mma.sync.m16n8k16 on the tensor cores\n\n");
    std::printf("      tensor core                            cpu reference\n");
    int bad = 0;
    double worst = 0.0;
    for (int m = 0; m < M; ++m) {
        std::printf("  ");
        for (int n = 0; n < N; ++n) std::printf("%7.1f", hD[m * N + n]);
        std::printf("     |");
        for (int n = 0; n < N; ++n) std::printf("%7.1f", hRef[m * N + n]);
        std::printf("\n");
        for (int n = 0; n < N; ++n) {
            const double e = std::fabs(hD[m * N + n] - hRef[m * N + n]);
            if (e > worst) worst = e;
            if (e > 1e-3) ++bad;
        }
    }

    std::printf("\nmax abs error %.3g over %d elements, %d mismatches\n",
                worst, M * N, bad);
    if (bad == 0) {
        std::printf("PASS -- the tensor cores computed this, and the fragment\n"
                    "layout in the header is correct.\n");
    } else {
        std::printf("FAIL -- if the output looks TRANSPOSED, check the B gather\n"
                    "(groupID selects B's column, not its row). If it looks\n"
                    "block-shuffled in 8-row chunks, check the +8 terms.\n");
    }

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dD));
    return bad == 0 ? 0 : 1;
}
