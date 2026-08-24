// Rung 2: FlashAttention — online softmax, tiled over K/V; S x S never materialized.

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cfloat>
#include <cmath>

namespace ladder {
namespace {

// #ifndef-guarded so scripts/sweep_attn_tiles.sh can override with -DKBR/-DKBC.
#ifndef KBR
#define KBR 64
#endif
#ifndef KBC
#define KBC 32  // sweep winner on GB10: 40 KB smem -> 2 resident blocks/SM
#endif

constexpr int kBr      = KBR;   // query rows per block
constexpr int kBc      = KBC;   // key/value columns per tile
constexpr int kMaxD    = 64;    // head dim this kernel is compiled for (exactly)
constexpr int kThreads = 128;   // 4 warps

// Compile-time loop bounds keep the accumulator in registers, not local memory.
constexpr int kThreadsPerRow = kThreads / kBr;             // 2 at defaults
constexpr int kAccPerThread  = kMaxD / kThreadsPerRow;     // 32 at defaults
constexpr int kColsPerThread = kBc / kThreadsPerRow;       // 32 at defaults

static_assert(kBr > 0 && kThreads % kBr == 0,
              "each query row needs a whole number of threads");
static_assert(kMaxD % kThreadsPerRow == 0 && kBc % kThreadsPerRow == 0,
              "row-sharing threads must split D and Bc evenly");

// Checked at build time against the GB10 99 KB per-block shared memory cap.
constexpr int kSmemBytes = (kBr * kMaxD          // Q tile
                          + kBc * kMaxD          // K tile
                          + kBc * kMaxD          // V tile
                          + kBr * kBc            // score tile
                           ) * static_cast<int>(sizeof(float));

static_assert(kSmemBytes <= gb10::kMaxSmemPerBlockBytes,
              "flash tile config exceeds the GB10 99KB per-block shared memory "
              "cap -- shrink Bc (or Br). Hopper's 228KB does not apply here.");

__global__ __launch_bounds__(kThreads)
void flash_tiled_kernel(int S, int D,
                        float scale, bool causal,
                        const float* __restrict__ Q,
                        const float* __restrict__ K,
                        const float* __restrict__ V,
                        float* __restrict__ O) {
    (void)D;   // the launcher guarantees D == kMaxD

    // Dynamic smem: 64 KB of static __shared__ would exceed the 48 KB static limit.
    extern __shared__ float smem[];
    float* Qs = smem;                       // kBr * kMaxD
    float* Ks = Qs + kBr * kMaxD;           // kBc * kMaxD
    float* Vs = Ks + kBc * kMaxD;           // kBc * kMaxD
    float* Ss = Vs + kBc * kMaxD;           // kBr * kBc

    const int bh   = blockIdx.y;
    const int row0 = blockIdx.x * kBr;
    const long long qkv_base = static_cast<long long>(bh) * S * kMaxD;

    const int r      = threadIdx.x / kThreadsPerRow;   // my query row in the tile
    const int half   = threadIdx.x % kThreadsPerRow;   // my output-column half
    const int col0   = half * kAccPerThread;
    const int i_glob = row0 + r;

    // m starts finite, not -INFINITY: -inf - (-inf) = NaN if the first tile is fully masked.
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc[kAccPerThread] = {0.0f};

    // Q tile loads once; out-of-range rows load 0.
    for (int idx = threadIdx.x; idx < kBr * kMaxD; idx += kThreads) {
        const int qr = idx / kMaxD, qd = idx % kMaxD;
        const int gq = row0 + qr;
        Qs[idx] = (gq < S) ? Q[qkv_base + static_cast<long long>(gq) * kMaxD + qd] : 0.0f;
    }

    // Causal: K/V tiles entirely above the diagonal are skipped, not masked.
    const int j_end   = causal ? min(S, row0 + kBr) : S;
    const int n_tiles = (j_end + kBc - 1) / kBc;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int j0 = tile * kBc;

        for (int idx = threadIdx.x; idx < kBc * kMaxD; idx += kThreads) {
            const int kr = idx / kMaxD, kd = idx % kMaxD;
            const int gj = j0 + kr;
            const bool in = (gj < S);
            const long long g = qkv_base + static_cast<long long>(gj) * kMaxD + kd;
            Ks[idx] = in ? K[g] : 0.0f;
            Vs[idx] = in ? V[g] : 0.0f;
        }
        __syncthreads();

        // Masked/out-of-range cells get a finite sentinel so they never win the row max.
        for (int c = half * kColsPerThread; c < (half + 1) * kColsPerThread; ++c) {
            const int j = j0 + c;
            float sv = -1e30f;
            if (j < S && !(causal && j > i_glob)) {
                float dot = 0.0f;
                #pragma unroll 16
                for (int d = 0; d < kMaxD; ++d)
                    dot += Qs[r * kMaxD + d] * Ks[c * kMaxD + d];
                sv = dot * scale;
            }
            Ss[r * kBc + c] = sv;
        }
        __syncthreads();   // each Ss row is written by 2 threads, read by both

        // Both threads of a row pair compute identical (m, l) from the shared row; no sync needed.
        float m_tile = -1e30f;
        for (int c = 0; c < kBc; ++c) m_tile = fmaxf(m_tile, Ss[r * kBc + c]);
        const float m_new = fmaxf(m_i, m_tile);
        const float alpha = expf(m_i - m_new);   // exp(old - new): always <= 1

        #pragma unroll
        for (int k = 0; k < kAccPerThread; ++k) acc[k] *= alpha;

        float row_sum = 0.0f;
        for (int c = 0; c < kBc; ++c) {
            const float p = expf(Ss[r * kBc + c] - m_new);
            row_sum += p;
            #pragma unroll
            for (int k = 0; k < kAccPerThread; ++k)
                acc[k] += p * Vs[c * kMaxD + col0 + k];
        }
        l_i = alpha * l_i + row_sum;   // alpha rescales BOTH acc and l
        m_i = m_new;

        __syncthreads();   // before Ks/Vs/Ss are overwritten next tile
    }

    // Out-of-range rows still reach every barrier above; they are masked only here.
    if (i_glob < S) {
        const float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < kAccPerThread; ++k)
            O[qkv_base + static_cast<long long>(i_glob) * kMaxD + col0 + k] =
                acc[k] * inv_l;
    }
}

}  // namespace

void launch_attn_flash_tiled(const AttnArgs& a, cudaStream_t s) {
    // Built for exactly D == kMaxD; refuse anything else loudly.
    if (a.D != kMaxD) {
        std::fprintf(stderr,
            "02_flash_tiled: compiled for D == %d, got D == %d. Rebuild with a "
            "matching kMaxD (see the 99KB sizing table in this file).\n",
            kMaxD, a.D);
        std::exit(EXIT_FAILURE);
    }

    static bool configured = false;   // >48KB dynamic smem needs a one-time opt-in
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(flash_tiled_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        configured = true;
    }

    dim3 grid(ceil_div(a.S, kBr), a.B * a.H);
    flash_tiled_kernel<<<grid, kThreads, kSmemBytes, s>>>(
        a.S, a.D, a.scale, a.causal, a.Q, a.K, a.V, a.O);
}

}  // namespace ladder
