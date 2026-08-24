// registry.cu — attention ladder registry; one entry per rung.

#include "kernels.cuh"

namespace ladder {

std::vector<AttnKernelSpec> build_attn_ladder() {
    std::vector<AttnKernelSpec> v;

    // No cuBLAS-equivalent library baseline exists for attention; rung 0 is the baseline.
    v.push_back({"00_naive_attention",
                 "3 kernels; materializes the full BxHxSxS score matrix in DRAM",
                 launch_attn_naive, Precision::FP32, 2e-4,
                 /*needs_scratch=*/true, AttnLadderStatus::naive});

    v.push_back({"01_fused_softmax",
                 "QK^T + softmax in one kernel; warp-shuffle row reductions",
                 launch_attn_fused_softmax, Precision::FP32, 2e-4,
                 /*needs_scratch=*/true, AttnLadderStatus::fused_softmax});

    v.push_back({"02_flash_tiled",
                 "online softmax, tiled over K/V; never materializes S x S",
                 launch_attn_flash_tiled, Precision::FP32, 2e-4,
                 /*needs_scratch=*/false, AttnLadderStatus::flash_tiled});

    // 2e-2 tolerance: bf16 carries only 8 mantissa bits.
    v.push_back({"03_flash_tensorcore",
                 "bf16 mma.sync m16n8k16 for both QK^T and P@V, fp32 accumulate",
                 launch_attn_flash_tc, Precision::BF16_TF32, 2e-2,
                 /*needs_scratch=*/false, AttnLadderStatus::flash_tc});

    return v;
}

}  // namespace ladder
