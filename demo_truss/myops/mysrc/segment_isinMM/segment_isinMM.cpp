#include <torch/extension.h>

// 声明 CUDA 函数
void segment_isinmm_cuda(
    const torch::Tensor u_clos,
    const torch::Tensor v_clos,
    const torch::Tensor uptr,
    const torch::Tensor vptr,
    torch::Tensor u_mask,
    torch::Tensor v_mask
);

// 包装函数
void segment_isinmm(
    torch::Tensor u_clos,
    torch::Tensor v_clos,
    torch::Tensor uptr,
    torch::Tensor vptr,
    torch::Tensor u_mask,
    torch::Tensor v_mask
) {
    // 调用 CUDA 函数
    segment_isinmm_cuda(u_clos, v_clos, uptr, vptr, u_mask, v_mask);
}

PYBIND11_MODULE(myisinmm, m) {
    m.def("segment_isinmm", &segment_isinmm, "Segment isin operation on CUDA");
}

