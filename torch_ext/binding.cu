// binding.cu — expose one ladder rung as torch.ops.kernel_ladder.sgemm (functional: never mutates the caller's C).

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include "sgemm/kernels.cuh"

// setup.py passes both with -D; the fallbacks keep a hand-run nvcc build working.
#ifndef LADDER_LAUNCH
#define LADDER_LAUNCH ladder::launch_vectorized
#endif
#ifndef LADDER_KERNEL_NAME
#define LADDER_KERNEL_NAME "05_vectorized"
#endif

namespace {

void check_inputs(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "kernel_ladder::sgemm: both inputs must be CUDA tensors "
                "(got a on ", a.device(), ", b on ", b.device(), "). "
                "There is no CPU implementation of this op on purpose -- a slow "
                "fallback would hide the fact that the custom kernel is not running.");
    TORCH_CHECK(a.device() == b.device(),
                "kernel_ladder::sgemm: inputs are on different devices (",
                a.device(), " vs ", b.device(), ")");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "kernel_ladder::sgemm: this op is fp32 only (got ",
                a.scalar_type(), " and ", b.scalar_type(), "). "
                "The fp32 kernels accumulate in fp32 from fp32 "
                "inputs; feeding them half/bfloat16 would be a different "
                "kernel with different numerics, so it gets a different op.");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "kernel_ladder::sgemm: expected 2-D matrices, got ",
                a.dim(), "-D and ", b.dim(), "-D. "
                "For batched or 3-D activations, reshape to 2-D first -- see "
                "bench_in_model.py, which folds [B, T, D] into [B*T, D].");
    TORCH_CHECK(a.size(1) == b.size(0),
                "kernel_ladder::sgemm: inner dimensions do not match: A is ",
                a.size(0), "x", a.size(1), ", B is ", b.size(0), "x", b.size(1));
    TORCH_CHECK(a.numel() > 0 && b.numel() > 0,
                "kernel_ladder::sgemm: empty input tensor");
    // Kernels index with int; silently truncating a >int32 dim would corrupt memory.
    TORCH_CHECK(a.size(0) <= INT32_MAX && b.size(1) <= INT32_MAX &&
                a.size(1) <= INT32_MAX,
                "kernel_ladder::sgemm: dimension exceeds int32; the kernels in "
                "src/sgemm/ index with int");
}

at::Tensor sgemm_cuda(const at::Tensor& a_in, const at::Tensor& b_in,
                      double alpha, double beta,
                      const c10::optional<at::Tensor>& c_in) {
    check_inputs(a_in, b_in);

    TORCH_CHECK(!a_in.requires_grad() && !b_in.requires_grad(),
                "kernel_ladder::sgemm has no backward pass, so it cannot be used "
                "on tensors that require grad. Wrap the call in torch.no_grad(), "
                "or use it in an inference-only module.");

    // Kernels assume plain row-major with no leading-dimension parameter; .contiguous() may copy.
    const at::Tensor a = a_in.contiguous();
    const at::Tensor b = b_in.contiguous();

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));

    const at::cuda::CUDAGuard guard(a.device());

    // zeros(), not empty(): the kernels read C unconditionally even when beta == 0.
    at::Tensor c;
    if (c_in.has_value() && c_in->defined()) {
        const at::Tensor& cv = *c_in;
        TORCH_CHECK(cv.is_cuda() && cv.scalar_type() == at::kFloat &&
                    cv.dim() == 2 && cv.size(0) == M && cv.size(1) == N,
                    "kernel_ladder::sgemm: C must be a 2-D fp32 CUDA tensor of "
                    "shape ", M, "x", N);
        // Clone, never alias: the op is functional.
        c = cv.contiguous().clone();
    } else {
        TORCH_CHECK(beta == 0.0,
                    "kernel_ladder::sgemm: beta=", beta, " was given with no C "
                    "tensor. beta scales an existing C, so passing one without "
                    "the other is a bug in the caller, not something to silently "
                    "treat as zero.");
        c = at::zeros({M, N}, a.options());
    }

    // Launch on PyTorch's current stream, not stream 0, to stay ordered with torch's own work.
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.device().index());

    ladder::GemmArgs args{
        M, N, K,
        static_cast<float>(alpha), static_cast<float>(beta),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        /*A_bf16=*/nullptr, /*B_bf16=*/nullptr,
        c.data_ptr<float>()
    };

    LADDER_LAUNCH(args, stream);

    // Checks the launch only; async execution faults still surface later.
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "kernel_ladder::sgemm (" LADDER_KERNEL_NAME ") launch failed: ",
                cudaGetErrorString(err),
                ". If this is cudaErrorInvalidConfiguration or "
                "OutOfResources, the kernel's tile size wants more shared memory "
                "or more threads per block than GB10 allows (99KB / 1024).");
    return c;
}

// Meta impl for torch.compile: shapes only, no storage; keep in step with sgemm_cuda.
at::Tensor sgemm_meta(const at::Tensor& a, const at::Tensor& b,
                      double /*alpha*/, double /*beta*/,
                      const c10::optional<at::Tensor>& /*c*/) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "kernel_ladder::sgemm: expected 2-D matrices");
    TORCH_CHECK(a.size(1) == b.size(0),
                "kernel_ladder::sgemm: inner dimensions do not match");
    return at::empty({a.size(0), b.size(1)}, a.options());
}

std::string kernel_name() { return std::string(LADDER_KERNEL_NAME); }

}  // namespace

// Schema has no (a!) annotations: the op mutates none of its arguments.
TORCH_LIBRARY(kernel_ladder, m) {
    m.def("sgemm(Tensor a, Tensor b, float alpha=1.0, float beta=0.0, "
          "Tensor? c=None) -> Tensor");
    // No dispatch key: one implementation for every backend.
    m.def("kernel_name() -> str", &kernel_name);
}

TORCH_LIBRARY_IMPL(kernel_ladder, CUDA, m) {
    m.impl("sgemm", TORCH_FN(sgemm_cuda));
}

TORCH_LIBRARY_IMPL(kernel_ladder, Meta, m) {
    m.impl("sgemm", TORCH_FN(sgemm_meta));
}

// Empty on purpose: import needs a PyInit_ symbol to load the .so, which runs the TORCH_LIBRARY constructors.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { (void)m; }
