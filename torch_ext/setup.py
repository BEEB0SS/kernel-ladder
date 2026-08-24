#!/usr/bin/env python3
"""Build the ladder's SGEMM kernel as a PyTorch custom op; pick the rung via KERNEL=<name>."""

import os
import sys

from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    sys.exit(
        "error: PyTorch is not importable, so its build helpers are not either.\n"
        "       A PyTorch build with sm_121 support is required on DGX Spark\n"
        "       (see the root README's requirements section)."
    )

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

# Rung 5 falls back to rung 4 for float4-misaligned shapes, so it must compile both.
KERNELS = {
    "00_naive":        (["00_naive.cu"],        "ladder::launch_naive"),
    "01_coalesced":    (["01_coalesced.cu"],    "ladder::launch_coalesced"),
    "02_smem_tiled":   (["02_smem_tiled.cu"],   "ladder::launch_smem_tiled"),
    "03_1d_blocktile": (["03_1d_blocktile.cu"], "ladder::launch_1d_blocktile"),
    "04_2d_blocktile": (["04_2d_blocktile.cu"], "ladder::launch_2d_blocktile"),
    "05_vectorized":   (["05_vectorized.cu", "04_2d_blocktile.cu"],
                        "ladder::launch_vectorized"),
    # 06_tensorcore is bf16-only: exposing it under the fp32 sgemm schema would
    # silently change arithmetic, so it needs its own opt-in op name.
}

KERNEL = os.environ.get("KERNEL", "05_vectorized")
if KERNEL not in KERNELS:
    sys.exit(f"error: KERNEL={KERNEL} is not one of: {', '.join(KERNELS)}")
kernel_srcs, launcher = KERNELS[KERNEL]

# Base sm_121 suffices on GB10 (mma.sync.m16n8k16 needs no 'a' suffix); requires CUDA 12.9+.
CUDA_ARCH = os.environ.get("LADDER_ARCH", "sm_121")

nvcc_flags = [
    "-O3",
    "-std=c++17",
    f"-arch={CUDA_ARCH}",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    # "bytes spill stores" in this output means the op will run slower than the standalone bench.
    "-Xptxas", "-v",
    f"-DLADDER_LAUNCH={launcher}",
    f'-DLADDER_KERNEL_NAME="{KERNEL}"',
]

# --use_fast_math is deliberately omitted: it would change numerics vs the C++ benchmark.

cxx_flags = ["-O3", "-std=c++17"]

# Default TORCH_CUDA_ARCH_LIST so an unset environment does not build for whatever GPU is local.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST",
                      "12.1" if CUDA_ARCH.startswith("sm_121") else "")

print(f"[kernel-ladder] binding rung {KERNEL} ({launcher}) for {CUDA_ARCH}")
print(f"[kernel-ladder] torch {torch.__version__}, "
      f"built against CUDA {torch.version.cuda}")

setup(
    name="kernel_ladder_torch",
    version="0.1.0",
    description="SGEMM kernels from kernel-ladder, exposed as PyTorch custom ops",
    ext_modules=[
        CUDAExtension(
            # Module name must match binding.cu's PYBIND11_MODULE and the __init__.py import.
            name="kernel_ladder_C",
            sources=[os.path.join(HERE, "binding.cu")]
                   + [os.path.join(SRC, "sgemm", s) for s in kernel_srcs],
            include_dirs=[SRC],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    py_modules=[],
    zip_safe=False,
)
