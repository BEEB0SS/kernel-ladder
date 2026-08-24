// registry.cu — Assembles the ladder. Adding a rung means adding one entry here.

#include "kernels.cuh"

namespace ladder {

std::vector<KernelSpec> build_ladder() {
    std::vector<KernelSpec> v;

    // Baseline must be first: the harness records its GFLOP/s for the speedup columns.
    v.push_back({"cublas", "vendor library, strict fp32 (the number to beat)",
                 launch_cublas, Precision::FP32, 2e-4, /*is_baseline=*/true, true});

    v.push_back({"00_naive", "1 thread per output; threadIdx.x indexes ROW (uncoalesced)",
                 launch_naive, Precision::FP32, 2e-4, false, LadderStatus::naive});

    v.push_back({"01_coalesced", "same math; threadIdx.x now indexes COLUMN",
                 launch_coalesced, Precision::FP32, 2e-4, false, LadderStatus::coalesced});

    v.push_back({"02_smem_tiled", "stage BMxBK / BKxBN tiles in shared memory",
                 launch_smem_tiled, Precision::FP32, 2e-4, false, LadderStatus::smem_tiled});

    v.push_back({"03_1d_blocktile", "TM outputs per thread; hoist the B load",
                 launch_1d_blocktile, Precision::FP32, 2e-4, false, LadderStatus::blocktile_1d});

    v.push_back({"04_2d_blocktile", "TM x TN register tile per thread",
                 launch_2d_blocktile, Precision::FP32, 2e-4, false, LadderStatus::blocktile_2d});

    v.push_back({"05_vectorized", "float4 loads + transposed A tile (no bank conflicts)",
                 launch_vectorized, Precision::FP32, 2e-4, false, LadderStatus::vectorized});

    // 2e-2 tolerance: bf16 inputs carry only 8 mantissa bits.
    v.push_back({"cublas_tf32", "vendor library, TF32 tensor cores (baseline for rung 6)",
                 launch_cublas_tf32, Precision::BF16_TF32, 2e-2, false, true});

    v.push_back({"06_tensorcore", "bf16 mma.sync m16n8k16, fp32 accumulate",
                 launch_tensorcore, Precision::BF16_TF32, 2e-2, false, LadderStatus::tensorcore});

    return v;
}

}  // namespace ladder
