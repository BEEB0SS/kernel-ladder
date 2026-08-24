// Attention ladder declarations: naive -> fused softmax -> flash tiled -> flash tensor core.
// GB10 (sm_121) caps shared memory at 99 KB per block; size tiles against that.

#pragma once

#include "attention_harness.cuh"

namespace ladder {

// Kernels stay in anonymous namespaces in their own .cu files; only launchers are exposed.
void launch_attn_naive        (const AttnArgs& a, cudaStream_t s);
void launch_attn_fused_softmax(const AttnArgs& a, cudaStream_t s);
void launch_attn_flash_tiled  (const AttnArgs& a, cudaStream_t s);
void launch_attn_flash_tc     (const AttnArgs& a, cudaStream_t s);

// Disabled kernels are reported as skipped rather than failed.
struct AttnLadderStatus {
    static constexpr bool naive         = true;
    static constexpr bool fused_softmax = true;
    static constexpr bool flash_tiled   = true;
    static constexpr bool flash_tc      = true;
};

std::vector<AttnKernelSpec> build_attn_ladder();

}  // namespace ladder
