// Rung 5: float4 loads + transposed A tile in shared memory.

#include "kernels.cuh"
#include "../common/check.cuh"

namespace ladder {
namespace {

// sweep_tiles.sh passes -DBM=... to every TU; undef so they don't smash the constexprs.
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
#ifdef TN
#undef TN
#endif

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int kThreads = (BM * BN) / (TM * TN);

// float4 chunks of each tile per thread.
constexpr int kAF4PerThread = (BM * BK) / (kThreads * 4);
constexpr int kBF4PerThread = (BK * BN) / (kThreads * 4);

static_assert(BK % 4 == 0, "BK must be a multiple of 4 for float4 loads of A");
static_assert(BN % 4 == 0, "BN must be a multiple of 4 for float4 loads of B");
static_assert(TM % 4 == 0 && TN % 4 == 0, "register tiles read as float4s");
static_assert((BM * BK) % (kThreads * 4) == 0,
              "A-tile must divide evenly among threads as float4s");
static_assert((BK * BN) % (kThreads * 4) == 0,
              "B-tile must divide evenly among threads as float4s");

__global__ __launch_bounds__(kThreads)
void vectorized_sgemm_kernel(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C) {
    __shared__ float As[BK * BM];      // transposed: [k][m], not [m][k]
    __shared__ float Bs[BK * BN];

    float acc[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];

    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Launcher gates K % 4 == 0; out-of-range lanes store zeros and still reach the barriers.
        #pragma unroll
        for (int t4 = 0; t4 < kAF4PerThread; ++t4) {
            const int f4     = threadIdx.x + t4 * kThreads;
            const int a_row  = f4 / (BK / 4);
            const int a_col4 = f4 % (BK / 4);
            const int ga_row = blockIdx.y * BM + a_row;
            const int ga_col = k0 + a_col4 * 4;
            const float4 t = (ga_row < M && ga_col + 3 < K)
                ? reinterpret_cast<const float4*>(&A[ga_row * K + ga_col])[0]
                : make_float4(0.f, 0.f, 0.f, 0.f);
            As[(a_col4 * 4 + 0) * BM + a_row] = t.x;
            As[(a_col4 * 4 + 1) * BM + a_row] = t.y;
            As[(a_col4 * 4 + 2) * BM + a_row] = t.z;
            As[(a_col4 * 4 + 3) * BM + a_row] = t.w;
        }
        #pragma unroll
        for (int t4 = 0; t4 < kBF4PerThread; ++t4) {
            const int f4     = threadIdx.x + t4 * kThreads;
            const int b_row  = f4 / (BN / 4);
            const int b_col4 = f4 % (BN / 4);
            const int gb_row = k0 + b_row;
            const int gb_col = blockIdx.x * BN + b_col4 * 4;
            reinterpret_cast<float4*>(&Bs[b_row * BN + b_col4 * 4])[0] =
                (gb_row < K && gb_col + 3 < N)
                    ? reinterpret_cast<const float4*>(&B[gb_row * N + gb_col])[0]
                    : make_float4(0.f, 0.f, 0.f, 0.f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = reinterpret_cast<const float4*>(
                    &As[k * BM + thread_row * TM + i])[0];
                regA[i + 0] = v.x; regA[i + 1] = v.y;
                regA[i + 2] = v.z; regA[i + 3] = v.w;
            }
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 v = reinterpret_cast<const float4*>(
                    &Bs[k * BN + thread_col * TN + j])[0];
                regB[j + 0] = v.x; regB[j + 1] = v.y;
                regB[j + 2] = v.z; regB[j + 3] = v.w;
            }
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
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const int c = blockIdx.x * BN + thread_col * TN + j;
                if (c + 3 < N) {
                    float4* out = reinterpret_cast<float4*>(&C[r * N + c]);
                    const float4 old = *out;
                    *out = make_float4(alpha * acc[i][j + 0] + beta * old.x,
                                       alpha * acc[i][j + 1] + beta * old.y,
                                       alpha * acc[i][j + 2] + beta * old.z,
                                       alpha * acc[i][j + 3] + beta * old.w);
                }
            }
        }
    }
}

}  // namespace

void launch_vectorized(const GemmArgs& a, cudaStream_t s) {
    // float4 paths need M, N, K multiples of 4; fall back to the scalar kernel otherwise.
    const bool vectorizable = (a.K % 4 == 0) && (a.N % 4 == 0) && (a.M % 4 == 0);
    if (!vectorizable) {
        launch_2d_blocktile(a, s);
        return;
    }
    dim3 block(kThreads);
    dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));
    vectorized_sgemm_kernel<<<grid, block, 0, s>>>(
        a.M, a.N, a.K, a.alpha, a.A_f32, a.B_f32, a.beta, a.C);
}

}  // namespace ladder
