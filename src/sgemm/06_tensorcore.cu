// Rung 6: bf16 tensor-core GEMM via mma.sync m16n8k16 (sm_121 has no wgmma/tcgen05).

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cuda_bf16.h>
#include <mma.h>

namespace ladder {
namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int kWarpsM  = 2;
constexpr int kWarpsN  = 2;
constexpr int kThreads = kWarpsM * kWarpsN * 32;

constexpr int kBM = WMMA_M * kWarpsM * 2;   // each warp owns a 32x32 quadrant
constexpr int kBN = WMMA_N * kWarpsN * 2;
constexpr int kBK = 16;

// Route B: raw mma.sync + ldmatrix; Route A (WMMA kernel below) kept as reference.
constexpr bool kRouteB   = true;
constexpr int  kBKB      = 32;              // deeper K-step: fewer barriers
constexpr int  kAStride  = kBKB + 8;        // +8 bf16 pad keeps ldmatrix rows off bank 0
constexpr int  kBStride  = kBN + 8;

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(unsigned (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}

__device__ __forceinline__
void mma_m16n8k16_bf16(float (&d)[4], const unsigned (&a)[4],
                       const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// cp.async 16-byte copy; src_size 0 zero-fills without reading the source.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__
void stage_ab_async(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* As, __nv_bfloat16* Bs,
                    int M, int N, int K, int k0,
                    int block_row0, int block_col0) {
    constexpr int kAChunks = kBKB / 8;
    constexpr int kBChunks = kBN / 8;
    for (int idx = threadIdx.x; idx < kBM * kAChunks; idx += kThreads) {
        const int r = idx / kAChunks, blk = idx % kAChunks;
        const int gr = block_row0 + r, gc = k0 + blk * 8;
        const bool valid = (gr < M && gc + 7 < K);
        cp_async16(&As[r * kAStride + blk * 8],
                   A + (valid ? static_cast<long long>(gr) * K + gc : 0), valid);
    }
    for (int idx = threadIdx.x; idx < kBKB * kBChunks; idx += kThreads) {
        const int r = idx / kBChunks, blk = idx % kBChunks;
        const int gr = k0 + r, gc = block_col0 + blk * 8;
        const bool valid = (gr < K && gc + 7 < N);
        cp_async16(&Bs[r * kBStride + blk * 8],
                   B + (valid ? static_cast<long long>(gr) * N + gc : 0), valid);
    }
}

__global__ __launch_bounds__(kThreads)
void tensorcore_sgemm_kernel_routeB(int M, int N, int K,
                                    float alpha,
                                    const __nv_bfloat16* __restrict__ A,
                                    const __nv_bfloat16* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C) {
    __shared__ __nv_bfloat16 As[2][kBM * kAStride];   // [stage][m][k], padded
    __shared__ __nv_bfloat16 Bs[2][kBKB * kBStride];  // [stage][k][n], padded

    const int lane    = threadIdx.x % 32;
    const int warp    = threadIdx.x / 32;
    const int warp_m  = warp / kWarpsN;
    const int warp_n  = warp % kWarpsN;
    const int groupID = lane / 4;
    const int tig     = lane % 4;

    const int block_row0 = blockIdx.y * kBM;
    const int block_col0 = blockIdx.x * kBN;
    const int n_steps    = (K + kBKB - 1) / kBKB;

    float acc[2][4][4] = {{{0.0f}}};   // [m-tile][n-tile][c0..c3]

    // Two-stage cp.async pipeline: slab t+1 streams in while slab t computes.
    stage_ab_async(A, B, As[0], Bs[0], M, N, K, 0, block_row0, block_col0);
    cp_async_commit();

    for (int step = 0; step < n_steps; ++step) {
        const int k0 = step * kBKB;
        const int st = step & 1;

        if (step + 1 < n_steps) {
            stage_ab_async(A, B, As[1 - st], Bs[1 - st], M, N, K,
                           k0 + kBKB, block_row0, block_col0);
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const __nv_bfloat16* Ast = As[st];
        const __nv_bfloat16* Bst = Bs[st];

        #pragma unroll
        for (int kk = 0; kk < kBKB; kk += 16) {
            // ldmatrix.x4 quadrant map: q = lane/8 -> (m-half q%2, k-half q/2).
            unsigned af[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int arow = warp_m * 32 + mt * 16
                               + ((lane / 8) % 2) * 8 + (lane % 8);
                const int acol = kk + (lane / 16) * 8;
                ldmatrix_x4(af[mt], &Ast[arow * kAStride + acol]);
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                // B stored [k][n]; .trans yields the column-major fragment .row.col wants.
                unsigned bf[2];
                const int brow = kk + ((lane % 16) / 8) * 8 + (lane % 8);
                const int bcol = warp_n * 32 + nt * 8;
                ldmatrix_x2_trans(bf, &Bst[brow * kBStride + bcol]);
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt)
                    mma_m16n8k16_bf16(acc[mt][nt], af[mt], bf, acc[mt][nt]);
            }
        }
        __syncthreads();
    }

    // M and N are multiples of 16, so every 16x8 tile is fully in or fully out.
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int r = blockIdx.y * kBM + warp_m * 32 + mt * 16
                            + groupID + ((c >= 2) ? 8 : 0);
                const int col = blockIdx.x * kBN + warp_n * 32 + nt * 8
                              + 2 * tig + (c & 1);
                if (r < M && col < N)
                    C[r * N + col] = alpha * acc[mt][nt][c] + beta * C[r * N + col];
            }
        }
    }
}

__global__ __launch_bounds__(kThreads)
void tensorcore_sgemm_kernel(int M, int N, int K,
                             float alpha,
                             const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             float beta,
                             float* __restrict__ C) {
    using namespace nvcuda::wmma;

    __shared__ __nv_bfloat16 As[kBM * kBK];
    __shared__ __nv_bfloat16 Bs[kBK * kBN];

    const int warp_id = threadIdx.x / 32;
    const int warp_m  = warp_id / kWarpsN;
    const int warp_n  = warp_id % kWarpsN;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
    // Bs [k][n] with ld = kBN is exactly WMMA's row_major matrix_b; no transpose needed.
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag[2];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += kBK) {
        // Launcher gates K % 16 == 0, so only the M/N edges need zero-fill.
        for (int idx = threadIdx.x; idx < kBM * kBK; idx += kThreads) {
            const int r = idx / kBK, c = idx % kBK;
            const int gr = blockIdx.y * kBM + r;
            As[idx] = (gr < M) ? A[gr * K + (k0 + c)] : __nv_bfloat16(0.f);
        }
        for (int idx = threadIdx.x; idx < kBK * kBN; idx += kThreads) {
            const int r = idx / kBN, c = idx % kBN;
            const int gc = blockIdx.x * kBN + c;
            Bs[idx] = (gc < N) ? B[(k0 + r) * N + gc] : __nv_bfloat16(0.f);
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 2; ++j)
            load_matrix_sync(b_frag[j], &Bs[warp_n * 32 + j * WMMA_N], kBN);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            load_matrix_sync(a_frag, &As[(warp_m * 32 + i * WMMA_M) * kBK], kBK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // M and N are multiples of 16, so every WMMA tile is fully in or fully out.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row0 = blockIdx.y * kBM + warp_m * 32 + i * WMMA_M;
            const int col0 = blockIdx.x * kBN + warp_n * 32 + j * WMMA_N;
            if (row0 < M && col0 < N) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                load_matrix_sync(c_frag, &C[row0 * N + col0], N, mem_row_major);
                for (int t = 0; t < c_frag.num_elements; ++t)
                    acc[i][j].x[t] = alpha * acc[i][j].x[t] + beta * c_frag.x[t];
                store_matrix_sync(&C[row0 * N + col0], acc[i][j], N, mem_row_major);
            }
        }
    }
}

}  // namespace

void launch_tensorcore(const GemmArgs& a, cudaStream_t s) {
    // Fall back at non-multiple-of-16 sizes so the harness still gets a correct answer.
    if (a.M % 16 || a.N % 16 || a.K % 16) {
        launch_vectorized(a, s);
        return;
    }
    dim3 block(kThreads);
    dim3 grid(ceil_div(a.N, WMMA_N * kWarpsN * 2),
              ceil_div(a.M, WMMA_M * kWarpsM * 2));

    // Consumes the harness's pre-converted bf16 copies so timing never includes conversion.
    if (kRouteB) {
        tensorcore_sgemm_kernel_routeB<<<grid, block, 0, s>>>(
            a.M, a.N, a.K, a.alpha, a.A_bf16, a.B_bf16, a.beta, a.C);
    } else {
        tensorcore_sgemm_kernel<<<grid, block, 0, s>>>(
            a.M, a.N, a.K, a.alpha, a.A_bf16, a.B_bf16, a.beta, a.C);
    }
}

}  // namespace ladder
