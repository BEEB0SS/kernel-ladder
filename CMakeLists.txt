cmake_minimum_required(VERSION 3.24)   # 3.24+ understands CUDA architecture 121
project(kernel_ladder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 121 = GB10 (sm_121); 121f adds block-scaled FP8/FP4 MMA, 121a adds sparse block-scaled MMA.
set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "CUDA arch (121, 121f, 121a)")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -Xptxas=-v reports per-kernel registers/smem/spills; -lineinfo lets ncu attribute stalls to lines.
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-v;-lineinfo;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>"
  "$<$<COMPILE_LANGUAGE:CXX>:-O2;-Wall;-Wextra>"
)

# --use_fast_math deliberately off: FTZ and relaxed div/sqrt would break agreement with the CPU oracle.

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS "12.9")
  message(FATAL_ERROR
    "CUDA ${CUDAToolkit_VERSION} found, but sm_121 (GB10) needs >= 12.9.\n"
    "DGX OS 7.5 ships CUDA 13.0.2. Check that PATH points at the right nvcc.")
endif()

add_executable(ladder
  src/sgemm/main.cu
  src/sgemm/registry.cu
  src/sgemm/cublas_ref.cu
  src/sgemm/00_naive.cu
  src/sgemm/01_coalesced.cu
  src/sgemm/02_smem_tiled.cu
  src/sgemm/03_1d_blocktile.cu
  src/sgemm/04_2d_blocktile.cu
  src/sgemm/05_vectorized.cu
  src/sgemm/06_tensorcore.cu
)
target_link_libraries(ladder PRIVATE CUDA::cublas CUDA::cudart)
set_target_properties(ladder PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  CUDA_SEPARABLE_COMPILATION OFF)   # whole-program compilation; better inlining

# Phase 2: fused attention, as a separate binary.
add_executable(attention
  src/attention/main.cu
  src/attention/registry.cu
  src/attention/00_naive_attention.cu
  src/attention/01_fused_softmax.cu
  src/attention/02_flash_tiled.cu
  src/attention/03_flash_tensorcore.cu
)
# cuBLAS linked only because src/common/check.cuh includes <cublas_v2.h> for CUBLAS_CHECK.
target_link_libraries(attention PRIVATE CUDA::cublas CUDA::cudart)
set_target_properties(attention PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  CUDA_SEPARABLE_COMPILATION OFF)

# Host-only tests; no GPU required.
add_executable(test_host tests/test_host.cpp)
set_target_properties(test_host PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(test_attention_host tests/test_attention_host.cpp)
set_target_properties(test_attention_host PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME host_tests      COMMAND test_host)
add_test(NAME attention_tests COMMAND test_attention_host)

message(STATUS "kernel-ladder: CUDA ${CUDAToolkit_VERSION}, arch sm_${CMAKE_CUDA_ARCHITECTURES}")
