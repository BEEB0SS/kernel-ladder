// kernels.cuh — launcher declarations for every rung of the SGEMM ladder.
// GB10 (sm_121) has no wgmma/tcgen05; rung 06 uses the Ampere mma.sync path.

#pragma once

#include "../common/harness.cuh"

namespace ladder {

// Kernels stay `static` inside their own .cu; only these launchers are exported.
void launch_naive          (const GemmArgs& a, cudaStream_t s);
void launch_coalesced      (const GemmArgs& a, cudaStream_t s);
void launch_smem_tiled     (const GemmArgs& a, cudaStream_t s);
void launch_1d_blocktile   (const GemmArgs& a, cudaStream_t s);
void launch_2d_blocktile   (const GemmArgs& a, cudaStream_t s);
void launch_vectorized     (const GemmArgs& a, cudaStream_t s);
void launch_tensorcore     (const GemmArgs& a, cudaStream_t s);
void launch_cublas         (const GemmArgs& a, cudaStream_t s);
void launch_cublas_tf32    (const GemmArgs& a, cudaStream_t s);

// Disabled kernels are reported as skipped rather than failed.
struct LadderStatus {
    static constexpr bool naive         = true;
    static constexpr bool coalesced     = true;
    static constexpr bool smem_tiled    = true;
    static constexpr bool blocktile_1d  = true;
    static constexpr bool blocktile_2d  = true;
    static constexpr bool vectorized    = true;
    static constexpr bool tensorcore    = true;
};

std::vector<KernelSpec> build_ladder();

// cuBLAS handle is created once (~100ms) so it never lands in a timed iteration.
void cublas_init();
void cublas_destroy();

}  // namespace ladder
