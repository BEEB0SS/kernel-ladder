// Rung 3: FlashAttention with both matmuls on mma.sync tensor cores — bf16 inputs, fp32 softmax.
// Tolerance 2e-2 is correct for bf16: the output is a convex combination of V rows,
// so error does not grow with S.

#include "kernels.cuh"
#include "../common/check.cuh"

#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

namespace ladder {
namespace {

constexpr int kBr      = 64;
constexpr int kBc      = 64;
constexpr int kMaxD    = 64;       // this kernel is compiled for exactly D == kMaxD
constexpr int kThreads = 128;      // 4 warps, each owning 16 query rows

// Option B chains the two mmas in registers; Option A round-trips P through smem.
constexpr bool kOptionB = true;

// +8 bf16 (16 B, alignment-preserving) row padding avoids 8-way ldmatrix bank conflicts.
constexpr int kStride      = kMaxD + 8;
constexpr int kWarps       = kThreads / 32;
constexpr int kRowsPerWarp = kBr / kWarps;
constexpr int kNTiles      = kBc / 8;                // score n-tiles per warp
constexpr int kKSteps      = kMaxD / 16;             // QK^T k-steps
constexpr int kPvKSteps    = kBc / 16;               // P@V k-steps
constexpr int kDTiles      = kMaxD / 8;              // P@V n-tiles

constexpr int kTileElems = kBc * kStride;   // one K or V stage
constexpr int kSmemBytes =
    (kBr * kStride                 // Q tile
   + 2 * kTileElems                // K tiles, double-buffered
   + 2 * kTileElems                // V tiles, double-buffered
   + kBr * kStride                 // P tile (Option A round trip)
    ) * static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kSmemBytes <= gb10::kMaxSmemPerBlockBytes,
              "flash-tc tile config exceeds the GB10 99KB per-block cap");
static_assert(kNTiles % 2 == 0,
              "Option B needs n-tiles paired (2i, 2i+1)");

// m16n8k16 .row.col: A is row-major (MxK), B column-major (KxN); c accumulates into d.
__device__ __forceinline__
void mma_m16n8k16_bf16(float (&d)[4], const unsigned (&a)[4],
                       const unsigned (&b)[2], const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// cvt.rn.bf16x2.f32 takes (hi, lo): the FIRST source becomes the HIGH half.
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    unsigned r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Each lane supplies one 16-byte row address; hardware distributes into the mma fragment layout.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(unsigned (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}

// src_size 0 zero-fills without reading, keeping out-of-range K/V rows zero.
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
void stage_kv_async(const __nv_bfloat16* __restrict__ K,
                    const __nv_bfloat16* __restrict__ V,
                    __nv_bfloat16* Ks, __nv_bfloat16* Vs,
                    long long base, int j0, int S) {
    constexpr int kChunks = kMaxD / 8;   // 16-byte chunks per row
    for (int idx = threadIdx.x; idx < kBc * kChunks; idx += kThreads) {
        const int row = idx / kChunks, blk = idx % kChunks;
        const int g = j0 + row;
        const bool valid = (g < S);
        const long long src =
            base + static_cast<long long>(valid ? g : 0) * kMaxD + blk * 8;
        cp_async16(&Ks[row * kStride + blk * 8], K + src, valid);
        cp_async16(&Vs[row * kStride + blk * 8], V + src, valid);
    }
}

__global__ __launch_bounds__(kThreads)
void flash_tc_kernel(int S, int D,
                     float scale, bool causal,
                     const __nv_bfloat16* __restrict__ Q,
                     const __nv_bfloat16* __restrict__ K,
                     const __nv_bfloat16* __restrict__ V,
                     float* __restrict__ O) {
    (void)D;   // the launcher guarantees D == kMaxD

    extern __shared__ __nv_bfloat16 tcsmem[];
    __nv_bfloat16* Qs = tcsmem;                    // kBr rows, stride kStride
    __nv_bfloat16* Ks = Qs + kBr * kStride;        // 2 stages of kBc rows
    __nv_bfloat16* Vs = Ks + 2 * kTileElems;       // 2 stages of kBc rows
    __nv_bfloat16* Ps = Vs + 2 * kTileElems;       // kBr rows (Option A only)

    const int lane    = threadIdx.x % 32;
    const int warp    = threadIdx.x / 32;
    const int groupID = lane / 4;
    const int tig     = lane % 4;

    const int bh    = blockIdx.y;
    const int row0  = blockIdx.x * kBr;
    const int wrow0 = row0 + warp * kRowsPerWarp;   // this warp's first query row
    const long long base = static_cast<long long>(bh) * S * kMaxD;

    for (int idx = threadIdx.x; idx < kBr * kMaxD; idx += kThreads) {
        const int r = idx / kMaxD, d = idx % kMaxD;
        const int g = row0 + r;
        Qs[r * kStride + d] =
            (g < S) ? Q[base + static_cast<long long>(g) * kMaxD + d]
                    : __nv_bfloat16(0.f);
    }
    __syncthreads();

    // Q fragments load once; ldmatrix.x4 matrix q = lane/8 covers quadrant (m-half q%2, k-half q/2).
    unsigned qa[kKSteps][4];
    #pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
        const int qrow = warp * kRowsPerWarp + ((lane / 8) % 2) * 8 + (lane % 8);
        const int qcol = ks * 16 + (lane / 16) * 8;
        ldmatrix_x4(qa[ks], &Qs[qrow * kStride + qcol]);
    }

    // Running softmax state for this lane's two rows (groupID, groupID+8).
    float m_run[2] = {-1e30f, -1e30f};
    float l_run[2] = {0.0f, 0.0f};
    float o[kDTiles][4] = {{0.0f}};

    const int j_end   = causal ? min(S, row0 + kBr) : S;
    const int n_tiles = (j_end + kBc - 1) / kBc;

    // Two-stage cp.async pipeline: tile t+1 streams in while tile t computes.
    stage_kv_async(K, V, Ks, Vs, base, 0, S);
    cp_async_commit();

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int j0 = tile * kBc;
        const int st = tile & 1;
        __nv_bfloat16* Kst = Ks + st * kTileElems;
        __nv_bfloat16* Vst = Vs + st * kTileElems;

        if (tile + 1 < n_tiles) {
            stage_kv_async(K, V, Ks + (1 - st) * kTileElems,
                           Vs + (1 - st) * kTileElems,
                           base, j0 + kBc, S);
            cp_async_commit();
            cp_async_wait<1>();   // tile t landed; t+1 still in flight
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // S_tile = Q @ K^T. K stored [key][dim] IS column-major B for .row.col — no transpose.
        float s[kNTiles][4];
        #pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt)
            #pragma unroll
            for (int c = 0; c < 4; ++c) s[nt][c] = 0.0f;

        #pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            #pragma unroll
            for (int ks = 0; ks < kKSteps; ++ks) {
                unsigned kb[2];
                const int krow = nt * 8 + (lane % 8);
                const int kcol = ks * 16 + ((lane % 16) / 8) * 8;
                ldmatrix_x2(kb, &Kst[krow * kStride + kcol]);
                mma_m16n8k16_bf16(s[nt], qa[ks], kb, s[nt]);
            }
        }

        // Scale, and mask out-of-range / above-diagonal cells with a finite sentinel (exp -> 0).
        #pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int mg = wrow0 + groupID + ((c >= 2) ? 8 : 0);
                const int jg = j0 + nt * 8 + 2 * tig + (c & 1);
                s[nt][c] = (jg < S && !(causal && jg > mg))
                         ? s[nt][c] * scale : -1e30f;
            }
        }

        // A row lives in 4 lanes x 2 cols: fold the own pair, then two XOR shuffles so all 4 lanes hold the max.
        float tmax[2] = {-1e30f, -1e30f};
        #pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            tmax[0] = fmaxf(tmax[0], fmaxf(s[nt][0], s[nt][1]));
            tmax[1] = fmaxf(tmax[1], fmaxf(s[nt][2], s[nt][3]));
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(0xffffffffu, tmax[h], 1));
            tmax[h] = fmaxf(tmax[h], __shfl_xor_sync(0xffffffffu, tmax[h], 2));
        }

        // Online softmax update, fp32 on CUDA cores.
        float m_new[2], alpha[2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            m_new[h] = fmaxf(m_run[h], tmax[h]);
            alpha[h] = expf(m_run[h] - m_new[h]);
        }
        #pragma unroll
        for (int nd = 0; nd < kDTiles; ++nd) {
            o[nd][0] *= alpha[0]; o[nd][1] *= alpha[0];
            o[nd][2] *= alpha[1]; o[nd][3] *= alpha[1];
        }

        float rsum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            s[nt][0] = expf(s[nt][0] - m_new[0]); rsum[0] += s[nt][0];
            s[nt][1] = expf(s[nt][1] - m_new[0]); rsum[0] += s[nt][1];
            s[nt][2] = expf(s[nt][2] - m_new[1]); rsum[1] += s[nt][2];
            s[nt][3] = expf(s[nt][3] - m_new[1]); rsum[1] += s[nt][3];
        }
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 1);
            rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 2);
            l_run[h] = alpha[h] * l_run[h] + rsum[h];
            m_run[h] = m_new[h];
        }

        // acc += P @ V. V stored [key][dim] is row-major in k — .trans loads.
        if (kOptionB) {
            // Two n-adjacent score tiles ARE the A fragment: one packing convert, no barrier, no smem.
            #pragma unroll
            for (int kt = 0; kt < kPvKSteps; ++kt) {
                const unsigned pa[4] = {
                    pack_bf16x2(s[2 * kt][0],     s[2 * kt][1]),
                    pack_bf16x2(s[2 * kt][2],     s[2 * kt][3]),
                    pack_bf16x2(s[2 * kt + 1][0], s[2 * kt + 1][1]),
                    pack_bf16x2(s[2 * kt + 1][2], s[2 * kt + 1][3]),
                };
                #pragma unroll
                for (int nd = 0; nd < kDTiles; ++nd) {
                    unsigned vb[2];
                    const int vrow = kt * 16 + ((lane % 16) / 8) * 8 + (lane % 8);
                    ldmatrix_x2_trans(vb, &Vst[vrow * kStride + nd * 8]);
                    mma_m16n8k16_bf16(o[nd], pa, vb, o[nd]);
                }
            }
        } else {
            // Option A: each warp writes and reads only its own 16 rows — __syncwarp() suffices.
            #pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                const int pr = warp * kRowsPerWarp + groupID;
                const int pc = nt * 8 + 2 * tig;
                Ps[pr * kStride + pc]           = __float2bfloat16(s[nt][0]);
                Ps[pr * kStride + pc + 1]       = __float2bfloat16(s[nt][1]);
                Ps[(pr + 8) * kStride + pc]     = __float2bfloat16(s[nt][2]);
                Ps[(pr + 8) * kStride + pc + 1] = __float2bfloat16(s[nt][3]);
            }
            __syncwarp();
            #pragma unroll
            for (int kt = 0; kt < kPvKSteps; ++kt) {
                unsigned pa[4];
                const int prow = warp * kRowsPerWarp + ((lane / 8) % 2) * 8 + (lane % 8);
                const int pcol = kt * 16 + (lane / 16) * 8;
                ldmatrix_x4(pa, &Ps[prow * kStride + pcol]);
                #pragma unroll
                for (int nd = 0; nd < kDTiles; ++nd) {
                    unsigned vb[2];
                    const int vrow = kt * 16 + ((lane % 16) / 8) * 8 + (lane % 8);
                    ldmatrix_x2_trans(vb, &Vst[vrow * kStride + nd * 8]);
                    mma_m16n8k16_bf16(o[nd], pa, vb, o[nd]);
                }
            }
        }
        __syncthreads();   // before Ks/Vs are overwritten next tile
    }

    // Divide by l once; write fp32 O through the C-fragment map.
    const float inv_l[2] = {1.0f / l_run[0], 1.0f / l_run[1]};
    #pragma unroll
    for (int nd = 0; nd < kDTiles; ++nd) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int mg = wrow0 + groupID + ((c >= 2) ? 8 : 0);
            const int dg = nd * 8 + 2 * tig + (c & 1);
            if (mg < S)
                O[base + static_cast<long long>(mg) * kMaxD + dg] =
                    o[nd][c] * inv_l[(c >= 2) ? 1 : 0];
        }
    }
}

}  // namespace

void launch_attn_flash_tc(const AttnArgs& a, cudaStream_t s) {
    // mma.sync needs S a multiple of 16 and this build needs D == kMaxD; fall back to rung 2.
    if (a.S % 16 != 0 || a.D != kMaxD) {
        launch_attn_flash_tiled(a, s);
        return;
    }

    if (kSmemBytes > 48 * 1024) {   // dynamic smem beyond the 48KB default needs an explicit opt-in
        static bool configured = false;
        if (!configured) {
            CUDA_CHECK(cudaFuncSetAttribute(flash_tc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
            configured = true;
        }
    }

    // Q/K/V are the harness's pre-converted bf16 copies; converting here would be timed.
    dim3 grid(ceil_div(a.S, kBr), a.B * a.H);
    flash_tc_kernel<<<grid, kThreads, kSmemBytes, s>>>(
        a.S, a.D, a.scale, a.causal, a.Q_bf16, a.K_bf16, a.V_bf16, a.O);
}

}  // namespace ladder
