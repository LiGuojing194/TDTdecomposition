#include <torch/extension.h>

__global__ void segmentIsinmmKernel(
        const int *u_clos, const int *v_clos,
        const int *uptr,
        const int *vptr, 
        bool *u_mask, bool *v_mask,
        const int numRows
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            int idxA = uptr[row];
            int endA = uptr[row + 1];
            int idxB = vptr[row];
            int endB = vptr[row + 1];

            while (idxA < endA && idxB < endB) {
            int colA = u_clos[idxA];
            int colB = v_clos[idxB];
            if (colA == colB) {
                u_mask[idxA] = true;
                v_mask[idxB] = true;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                idxA++;
            } else {
                idxB++;
            }
            }
        }
    }


void segment_isinmm_cuda(
    const torch::Tensor u_clos,
    const torch::Tensor v_clos,
    const torch::Tensor uptr,
    const torch::Tensor vptr,
    torch::Tensor u_mask,
    torch::Tensor v_mask
) {
    const int numRows = uptr.size(0) - 1;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    segmentIsinmmKernel<<<blocks, threads>>>(
        u_clos.data_ptr<int>(),
        v_clos.data_ptr<int>(),
        uptr.data_ptr<int>(),
        vptr.data_ptr<int>(),
        u_mask.data_ptr<bool>(),
        v_mask.data_ptr<bool>(),
        numRows
    );
}
