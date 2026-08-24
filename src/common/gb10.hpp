// gb10.hpp — hardware constants for the NVIDIA DGX Spark (GB10 Grace Blackwell).

#pragma once

namespace gb10 {

constexpr double kDramBandwidthPeakGBs     = 273.0;  // spec sheet, 256-bit LPDDR5X @ 8533 MT/s
constexpr double kDramBandwidthMeasuredGBs = 231.0;  // saturating-read kernel, 8 GiB buffer

// Derived (6144 CUDA cores x 2 FLOP x clock), not published; verify with measure_ceilings.sh.
constexpr double kFp32PeakTFLOPS = 29.7;

// Measured (mmapeak); FP8 == BF16 rate on GB10 — no 2x FP8 speedup as on datacenter parts.
constexpr double kTf32TensorTFLOPS = 53.3;
constexpr double kBf16TensorTFLOPS = 212.9;
constexpr double kFp16TensorTFLOPS = 213.0;
constexpr double kFp8TensorTFLOPS  = 213.7;
constexpr double kFp4DenseTFLOPS   = 427.0;

constexpr int kNumSMs            = 48;
constexpr int kCudaCoresPerSM    = 128;
constexpr int kL2CacheBytes      = 24 * 1024 * 1024;

// GB10 caps a single block at 99 KB smem (Hopper allows 228 KB); size tiles against 99 KB.
constexpr int kMaxSmemPerBlockBytes = 99 * 1024;
constexpr int kMaxSmemPerSMBytes    = 100 * 1024;

constexpr int kWarpSize            = 32;
constexpr int kMaxThreadsPerBlock  = 1024;

// Ridge point: FLOP/byte at which a kernel flips from memory-bound to compute-bound.
constexpr double kRidgePointFp32 = (kFp32PeakTFLOPS   * 1e12) / (kDramBandwidthMeasuredGBs * 1e9);
constexpr double kRidgePointBf16 = (kBf16TensorTFLOPS * 1e12) / (kDramBandwidthMeasuredGBs * 1e9);

}  // namespace gb10
