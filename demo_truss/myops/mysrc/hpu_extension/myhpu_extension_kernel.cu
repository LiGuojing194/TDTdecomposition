#include <torch/extension.h>
#include <iostream>

//********************************segment add********************************************************//
__global__ void segmentAddKernel(const int *col_indices, const int *col_ptr, int *output, const int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int start = col_ptr[row];
        int end = col_ptr[row + 1];
        if (start < end){
            int sum = col_indices[start];
            start += 1;
            while(start < end){
                sum += col_indices[start];
                start += 1;
            }
            output[row] = sum;
        }
    }
}

void segment_add_cuda(const int *col_indices, const int *col_ptr, int *output, const int numRows) {
    dim3 block(512);
    dim3 grid((numRows + block.x - 1) / block.x);
    segmentAddKernel<<<grid, block>>>(col_indices, col_ptr, output, numRows);
}
//********************************two ptr segment isin********************************************************//
__global__ void checkElementsTwoPtr(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int col = b_col_indices[row];
        int idxA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[row];
        int idxB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;

        while (idxA < endA && idxB < endB) {
            int colA = b_col_indices[idxA];
            int colB = b_col_indices[idxB];
            if (colA == colB) {
                atomicAdd(&output[idxA], 1);
                atomicAdd(&output[idxB], 1);
                count++;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                idxA++;
            } else {
                idxB++;
            }
        }
        atomicAdd(&output[row], count);
    }
}

void checkElementsTwoPtr_cuda(
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    torch::Tensor output
) {
    const int numRows = output.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    checkElementsTwoPtr<<<blocks, threads>>>(
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        output.data_ptr<int>(),
        numRows
    );
}
//****************************two segment isin tile*************************************************************//
__global__ void checkElementsTwoPtrTile(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut, 
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int i = row / n_cut;
        int j = row % n_cut;
        int col = b_col_indices[i]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[i]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;

        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                atomicAdd(&output[startA], 1);
                atomicAdd(&output[startB], 1);
                count++;
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        atomicAdd(&output[i], count);
    }
}

void checkElementsTwoPtrTile_cuda(
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut, 
    torch::Tensor output
) {
    const int numRows = output.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    checkElementsTwoPtrTile<<<blocks, threads>>>(
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        n_cut,
        output.data_ptr<int>(),
        numRows
    );
    // 检查 CUDA 错误
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }
}

//****************************sub affect_support *************************************************************//
__global__ void AllAffectedSupport(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int e = e_affect[row];
        int col = b_col_indices[e];
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e];
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && (mark[e] || mark[startA] || mark[startB])){
                    if (output[startA]>l){
                        atomicSub(&output[startA], 1);
                        n_mark[startA] = true;
                    }
                    if (output[startB]>l){
                        atomicSub(&output[startB], 1);
                        n_mark[startB] = true;
                    }
                    if (!mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            atomicSub(&output[e], count);
            n_mark[e] = true;
        }
    }
}

void AllAffectedSupport_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
) {
    const int numRows = e_affect.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        numRows
    );
}  
//////////////////*******************not***********************************************
__global__ void AllAffectedSupport_not(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int e = e_affect[row];
        int col = b_col_indices[e];
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e];
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && !(mark[e] && mark[startA] && mark[startB])){
                    if (output[startA]>l){
                        int temp = atomicSub(&output[startA], 1);
                        if ((temp-1) == l){
                            n_mark[startA] = true;
                        }
                    }
                    if (output[startB]>l){
                        int temp = atomicSub(&output[startB], 1);
                        if ((temp-1) == l){
                            n_mark[startB] = true;
                        }
                    }
                    if (mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            int temp = atomicSub(&output[e], count);
            if ((temp - count) <= l){
                n_mark[e] = true;
            }
        }
    }
}

void AllAffectedSupport_not_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
) {
    const int numRows = e_affect.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport_not<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        numRows
    );
}  
//////////////////*******************not windows***********************************************
__global__ void AllAffectedSupport_notwin(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    const int l, bool * n_mark,
    int *output, const int windows_high, bool *in_windows_mask, int *buff_index, int *buf_size, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int e = e_affect[row];
        int col = b_col_indices[e];
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e];
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && !(mark[e] && mark[startA] && mark[startB])){
                    if (output[startA]>l){
                        int temp = atomicSub(&output[startA], 1);
                        if ((temp-1) == l){
                            n_mark[startA] = true;
                        }
                        if(temp == windows_high){
                            in_windows_mask[startA] = true;
                            int insert_idx = atomicAdd(buf_size, 1);
                            buff_index[insert_idx] = startA;
                        }
                    }
                    if (output[startB]>l){
                        int temp = atomicSub(&output[startB], 1);
                        if ((temp-1) == l){
                            n_mark[startB] = true;
                        }
                        if(temp == windows_high){
                            in_windows_mask[startB] = true;
                            int insert_idx = atomicAdd(buf_size, 1);
                            buff_index[insert_idx] = startB;
                        }
                    }
                    if (mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            int temp = atomicSub(&output[e], count);
            if ((temp - count) <= l){
                n_mark[e] = true;
            }
            if(temp - count < windows_high && temp >= windows_high){
                in_windows_mask[e] = true;
                int insert_idx = atomicAdd(buf_size, 1);
                buff_index[insert_idx] = e;
            }
        }
    }
}

void AllAffectedSupport_notwin_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    const int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
) {
    const int numRows = e_affect.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport_notwin<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        windows_high,
        in_windows_mask.data_ptr<bool>(),
        buff_index.data_ptr<int>(),
        buf_size.data_ptr<int>(),
        numRows
    );
}  
//****************************sub affect_support tile*************************************************************//
__global__ void AllAffectedSupport_tile(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int i = row / n_cut;
        int j = row  % n_cut;
        int e = e_affect[i];
        int col = b_col_indices[e]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && (mark[e] || mark[startA] || mark[startB])){
                    if (output[startA]>l){
                    atomicSub(&output[startA], 1);
                    n_mark[startA] = true;
                    }
                    if (output[startB]>l){
                        atomicSub(&output[startB], 1);
                        n_mark[startB] = true;
                    }
                    if (!mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
                atomicSub(&output[e], count);
                n_mark[e] = true;
        }
    }
}

void AllAffectedSupport_tile_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
) {
    const int numRows = e_affect.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport_tile<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        n_cut,
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        numRows
    );
    // 检查 CUDA 错误
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }
}  
//////////////////*******************not***********************************************
__global__ void AllAffectedSupport_tilenot(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    const int l, bool * n_mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int i = row / n_cut;
        int j = row  % n_cut;
        int e = e_affect[i];
        int col = b_col_indices[e]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && !(mark[e] && mark[startA] && mark[startB])){
                    if (output[startA]>l){
                    int temp = atomicSub(&output[startA], 1);
                    if ((temp-1) == l){
                            n_mark[startA] = true;
                        }
                    }
                    if (output[startB]>l){
                        int temp = atomicSub(&output[startB], 1);
                        if ((temp-1) == l){
                                n_mark[startB] = true;
                            }
                    }
                    if (mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            int temp = atomicSub(&output[e], count);
            if ((temp - count) <= l){
                n_mark[e] = true;
            }
        }
    }
}

void AllAffectedSupport_tilenot_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output
) {
    const int numRows = e_affect.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport_tilenot<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        n_cut,
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        numRows
    );
}
//////////////////*******************not windows***********************************************
__global__ void AllAffectedSupport_tilenotwin(
    const int *e_affect, 
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    const int l, bool * n_mark,
    int *output, const int windows_high, bool *in_windows_mask, int *buff_index, int *buf_size, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int i = row / n_cut;
        int j = row  % n_cut;
        int e = e_affect[i];
        int col = b_col_indices[e]*n_cut + j;
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[e]*n_cut + j;
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (colA != -1 && !(mark[e] && mark[startA] && mark[startB])){
                    if (output[startA]>l){
                        int temp = atomicSub(&output[startA], 1);
                        if ((temp-1) == l){
                            n_mark[startA] = true;
                        }
                        if(temp == windows_high){
                            in_windows_mask[startA] = true;
                            int insert_idx = atomicAdd(buf_size, 1);
                            buff_index[insert_idx] = startA;
                        }
                    }
                    if (output[startB]>l){
                        int temp = atomicSub(&output[startB], 1);
                        if ((temp-1) == l){
                                n_mark[startB] = true;
                        }
                        if(temp == windows_high){
                            in_windows_mask[startB] = true;
                            int insert_idx = atomicAdd(buf_size, 1);
                            buff_index[insert_idx] = startB;
                        }
                    }
                    if (mark[e]){
                        count++;
                    }  
                }
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0){
            int temp = atomicSub(&output[e], count);
            if ((temp - count) <= l){
                    n_mark[e] = true;
            }
            if(temp - count < windows_high && temp >= windows_high){
                in_windows_mask[e] = true;
                int insert_idx = atomicAdd(buf_size, 1);
                buff_index[insert_idx] = e;
            }
        }
    }
}

void AllAffectedSupport_tilenotwin_cuda(
    const torch::Tensor e_affect,
    const torch::Tensor a_rows,
    const torch::Tensor b_col_indices,
    const torch::Tensor b_row_ptr,
    const int n_cut,
    const torch::Tensor mark,
    const int l, 
    torch::Tensor n_mark, 
    torch::Tensor output,
    const int windows_high, 
    torch::Tensor in_windows_mask, 
    torch::Tensor buff_index, 
    torch::Tensor buf_size
) {
    const int numRows = e_affect.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    AllAffectedSupport_tilenotwin<<<blocks, threads>>>(
        e_affect.data_ptr<int>(),
        a_rows.data_ptr<int>(),
        b_col_indices.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        n_cut,
        mark.data_ptr<bool>(),
        l,
        n_mark.data_ptr<bool>(),
        output.data_ptr<int>(),
        windows_high,
        in_windows_mask.data_ptr<bool>(),
        buff_index.data_ptr<int>(),
        buf_size.data_ptr<int>(),
        numRows
    );
}    
//********************************two ptr segment isin mask1 mask2********************************************************//
__global__ void checkElementsTwoPtrIsInIs(
    const int *a_columns, const int *a_row_ptr,
    const int *b_columns, const int *b_row_ptr,
    bool *mask1, bool *mask2, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int idxA = a_row_ptr[row];
        int endA = a_row_ptr[row+1];
        int idxB = b_row_ptr[row];
        int endB = b_row_ptr[row+ 1];

        while (idxA < endA && idxB < endB) {
            int colA = a_columns[idxA];
            int colB = b_columns[idxB];
            if (colA == colB && colA != -1) {
                mask1[idxA] = true;
                mask2[idxB] = true;
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

void segment_isinis_m_cuda(
    const torch::Tensor a_columns,
    const torch::Tensor a_row_ptr,
    const torch::Tensor b_columns,
    const torch::Tensor b_row_ptr,
    torch::Tensor mask1,
    torch::Tensor mask2
) {
    const int numRows = a_row_ptr.size(0)-1;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    checkElementsTwoPtrIsInIs<<<blocks, threads>>>(
        a_columns.data_ptr<int>(),
        a_row_ptr.data_ptr<int>(),
        b_columns.data_ptr<int>(),
        b_row_ptr.data_ptr<int>(),
        mask1.data_ptr<bool>(),
        mask2.data_ptr<bool>(),
        numRows
    );
}
//*******************************sub_suppport_affect****************************//
__global__ void checkTriangleSupportSub(
    const int *left_nbr, const int *right_nbr,
    const int *e_curr_rep, int *support,  const bool *mask1, const bool *mask2,
    bool *e_affect_mask, const int l, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int left = left_nbr[row];
        int right = right_nbr[row];
        bool left_s = mask1[row];
        bool right_s = mask2[row];
        int curr = e_curr_rep[row];
        if(left_s && right_s){
            atomicSub(&support[left], 1);
            atomicSub(&support[right], 1);
            if (!e_affect_mask[left]) {
                e_affect_mask[left] = true;
            }
            if (!e_affect_mask[right]) {
                e_affect_mask[right] = true;
            }
        }
        else if(left_s && !right_s){
            if (curr<right){
                atomicSub(&support[left], 1);
                if (!e_affect_mask[left]) {
                    e_affect_mask[left] = true;
                }
            }
        }
        else if(!left_s && right_s){
            if (curr<left){
                atomicSub(&support[right], 1);
                if (!e_affect_mask[right]) {
                    e_affect_mask[right] = true;
                }
            }
        }
    }
}

void sub_suppport_affect_cuda(
    const torch::Tensor left_nbr,
    const torch::Tensor right_nbr,
    const torch::Tensor e_curr_rep,
    torch::Tensor support,
    const torch::Tensor mask1, 
    const torch::Tensor mask2,
    torch::Tensor e_affect_mask,
    const int l
){
    const int numRows = left_nbr.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);

    checkTriangleSupportSub<<<blocks, threads>>>(
        left_nbr.data_ptr<int>(),
        right_nbr.data_ptr<int>(),
        e_curr_rep.data_ptr<int>(),
        support.data_ptr<int>(),
        mask1.data_ptr<bool>(),
        mask2.data_ptr<bool>(),
        e_affect_mask.data_ptr<bool>(),
        l,
        numRows
    );
}

//*******************************peeling_undirect*********************************************// 让l传l+1吧
__global__ void peeling_edges_undirect(
    const int *e_curr, const int *rows,
    const int *columns, const int *columns_g, const int *row_ptr,  const int *edges_id_nbr, int *support, const bool *e_mask,
    bool *n_mark, const bool* in_curr, const int l, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int e = e_curr[i];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row];
        int endA = row_ptr[row+1];
        int idxB = row_ptr[col];
        int endB = row_ptr[col+1];
        while (idxA < endA && idxB < endB) {
            int colA = columns_g[idxA];
            int colB = columns_g[idxB];
            if (colA == colB ) {
                int e2 = edges_id_nbr[idxA];
                int e3 = edges_id_nbr[idxB];
                if (e_mask[e2] && e_mask[e3]){
                    bool f2 = in_curr[e2];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[e2], 1);
                        if (s == l){
                            n_mark[e2] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        if (e<e3){
                            int s = atomicSub(&support[e2], 1);
                            if (s == l){
                                n_mark[e2] = true;}
                        }
                    }
                    else if(f2 && !f3){
                        if (e<e2){
                            int s = atomicSub(&support[e3], 1);
                            if (s == l){
                                n_mark[e3] = true;}
                        }
                    }
                }
                idxA++;
                idxB++;
            } else if (colA < colB) {
                // if (colB > columns_g[endA-1]){
                // break;
                // }
                idxA++;
            } else {
                // if (colA > columns_g[endB-1]){
                //     break;
                // }
                idxB++;
            }
        }
    }
}

void peeling_undirect_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor columns_g,
    const torch::Tensor row_ptr,
    const torch::Tensor edges_id_nbr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l
){
    const int numRows = e_curr.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l += 1;

    peeling_edges_undirect<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        columns_g.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        edges_id_nbr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        numRows
    );   
}

//*******************************peeling_undirect_tile*********************************************//
__global__ void peeling_edges_undirect_tile(
    const int *e_curr, const int *rows,
    const int *columns, const int *columns_g, const int *row_ptr,  const int *edges_id_nbr, int *support, const bool *e_mask,
    bool *n_mark, const bool* in_curr, const int l, const int n_cut, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int j = i / n_cut;
        int k = i  % n_cut;
        int e = e_curr[j];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row*n_cut+k];
        int endA = row_ptr[row*n_cut+k+1];
        int idxB = row_ptr[col*n_cut+k];
        int endB = row_ptr[col*n_cut+k+1];   
        while (idxA < endA && idxB < endB) {
            int colA = columns_g[idxA];
            int colB = columns_g[idxB];
            if (colA == colB) {  //难以在columns_g上标记上-1，或者分别传e_mask和in_curr; 或者这里的判断变成support[e2]>=l && support[e3]>=l
                // if (colA != -1){
                int e2 = edges_id_nbr[idxA];
                int e3 = edges_id_nbr[idxB];
                if (e_mask[e2] && e_mask[e3]){
                    bool f2 = in_curr[e2];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[e2], 1);
                        if (s == l){
                            n_mark[e2] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        if (e<e3){
                            int s = atomicSub(&support[e2], 1);
                            if (s == l){
                                n_mark[e2] = true;}
                        }
                    }
                    else if(f2 && !f3){
                        if (e<e2){
                            int s = atomicSub(&support[e3], 1);
                            if (s == l){
                                n_mark[e3] = true;}
                        }
                    }
                }
                idxA++;
                idxB++;
            } else if (colA < colB) {
                // if (colB > columns_g[endA-1]){
                // break;
                // }
                idxA++;
            } else {
                // if (colA > columns_g[endB-1]){
                //     break;
                // }
                idxB++;
            }
        }
    }
}
void peeling_undirect_tile_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor columns_g,
    const torch::Tensor row_ptr,
    const torch::Tensor edges_id_nbr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut
){
    const int numRows = e_curr.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l += 1;

    peeling_edges_undirect_tile<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        columns_g.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        edges_id_nbr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        n_cut,
        numRows
    );   
}

/*******************************************************peeling_direct_oo*********************************************/
__global__ void peeling_edges_direct_oo(
    const int *e_curr, const int *rows,
    const int *columns, const int *row_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool *in_curr, const int l, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int e = e_curr[i];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row];
        int endA = row_ptr[row+1];
        int idxB = row_ptr[col];
        int endB = row_ptr[col+1];
        while (idxA < endA && idxB < endB) {
            int colA = columns[idxA];
            int colB = columns[idxB];
            if (colA == colB ) {
               //e2就是idxA, e3就是idxB, e<e2 && e<e3
                if (e_mask[idxA] && e_mask[idxB]){
                    bool f2 = in_curr[idxA];
                    bool f3 = in_curr[idxB];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;
                        }
                        s = atomicSub(&support[idxB], 1);
                        if (s == l){
                            n_mark[idxB] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;}
                    }
                    else if(f2 && !f3){
                        int s = atomicSub(&support[idxB], 1);
                        if (s == l){
                            n_mark[idxB] = true;}
                    }
                }
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

void peeling_direct_oo_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l){
    const int numRows = e_curr.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l +=  1;

    peeling_edges_direct_oo<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        numRows
    );   
}

/*******************************************************peeling_direct_oo_tile *********************************************/
__global__ void peeling_edges_direct_tile_oo(
    const int *e_curr, const int *rows,
    const int *columns, const int *row_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool* in_curr, const int l, const int n_cut, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int j = i / n_cut;
        int k = i  % n_cut;
        int e = e_curr[j];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row*n_cut+k];
        int endA = row_ptr[row*n_cut+k+1];
        int idxB = row_ptr[col*n_cut+k];
        int endB = row_ptr[col*n_cut+k+1];  
        while (idxA < endA && idxB < endB) {
            int colA = columns[idxA];
            int colB = columns[idxB];
            if (colA == colB ) {
               //e2就是idxA, e3就是idxB, e<e2 && e<e3
                if (e_mask[idxA] && e_mask[idxB]){
                    bool f2 = in_curr[idxA];
                    bool f3 = in_curr[idxB];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;
                        }
                        s = atomicSub(&support[idxB], 1);
                        if (s == l){
                            n_mark[idxB] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;}
                    }
                    else if(f2 && !f3){
                        int s = atomicSub(&support[idxB], 1);
                        if (s == l){
                            n_mark[idxB] = true;}
                    }
                }
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

void peeling_direct_tile_oo_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut){
    const int numRows = e_curr.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l += 1;

    peeling_edges_direct_tile_oo<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        n_cut,
        numRows
    );   
}

/*******************************************************peeling_direct_ii*********************************************/
__global__ void peeling_edges_direct_ii(
    const int *e_curr, const int *rows,
    const int *columns, const int *r_edges, const int *re_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool *in_curr, const int l, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int e = e_curr[i];
        int row = rows[e];
        int col = columns[e];
        int idxA = re_ptr[row];
        int endA = re_ptr[row+1];
        int idxB = re_ptr[col];
        int endB = re_ptr[col+1];
        while (idxA < endA && idxB < endB) {
            int e2 = r_edges[idxA];
            int e3 = r_edges[idxB];
            int colA = rows[e2];
            int colB = rows[e3];
            if (colA == colB ) {
               //e2<e1 && e3<e1
                if (e_mask[e2] && e_mask[e3]){
                    bool f2 = in_curr[e2];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[e2], 1);
                        if (s == l){
                            n_mark[e2] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    // else if(!f2 && f3){
                    //     if (e<e3){
                    //         int s = atomicSub(&support[e2], 1);
                    //         if (s == l){
                    //             n_mark[e2] = true;}
                    //     }
                    // }
                    // else if(f2 && !f3){
                    //     if (e<e2){
                    //         int s = atomicSub(&support[e3], 1);
                    //         if (s == l){
                    //             n_mark[e3] = true;}
                    //     }
                    // }
                }
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

void peeling_direct_ii_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l){
    const int numRows = e_curr.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l +=  1;
    peeling_edges_direct_ii<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        r_edges.data_ptr<int>(),
        re_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        numRows
    ); 
}

/*******************************************************peeling_direct_tile_ii*********************************************/
__global__ void peeling_edges_direct_tile_ii(
    const int *e_curr, const int *rows,
    const int *columns, const int *r_edges, const int *re_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool *in_curr, const int l, const int n_cut, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int j = i / n_cut;
        int k = i  % n_cut;
        int e = e_curr[j];
        int row = rows[e];
        int col = columns[e];
        int idxA = re_ptr[row*n_cut+k];
        int endA = re_ptr[row*n_cut+k+1];
        int idxB = re_ptr[col*n_cut+k];
        int endB = re_ptr[col*n_cut+k+1];  
        while (idxA < endA && idxB < endB) {
            int e2 = r_edges[idxA];
            int e3 = r_edges[idxB];
            int colA = rows[e2];
            int colB = rows[e3];
            if (colA == colB ) {
               //e2<e1 && e3<e1
                if (e_mask[e2] && e_mask[e3]){
                    bool f2 = in_curr[e2];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[e2], 1);
                        if (s == l){
                            n_mark[e2] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    // else if(!f2 && f3){
                    //     if (e<e3){
                    //         int s = atomicSub(&support[e2], 1);
                    //         if (s == l){
                    //             n_mark[e2] = true;}
                    //     }
                    // }
                    // else if(f2 && !f3){
                    //     if (e<e2){
                    //         int s = atomicSub(&support[e3], 1);
                    //         if (s == l){
                    //             n_mark[e3] = true;}
                    //     }
                    // }
                }
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

void peeling_direct_tile_ii_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut){
    const int numRows = e_curr.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l +=  1;
    peeling_edges_direct_tile_ii<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        r_edges.data_ptr<int>(),
        re_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        n_cut,
        numRows
    ); 
}

/*******************************************************peeling_direct_oi*********************************************/
__global__ void peeling_edges_direct_oi(
    const int *e_curr, const int *rows,
    const int *columns, const int *row_ptr, const int *r_edges, const int *re_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool *in_curr, const int l, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int e = e_curr[i];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row];
        int endA = row_ptr[row+1];
        int idxB = re_ptr[col];
        int endB = re_ptr[col+1];
        while (idxA < endA && idxB < endB) {
            int colA = columns[idxA];
            int e3 = r_edges[idxB];
            int colB = rows[e3];
            if (colA == colB ) {
               //idxA是e1, e2<e1<e3
                if (e_mask[idxA] && e_mask[e3]){
                    bool f2 = in_curr[idxA];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;}
                    }
                    // else if(f2 && !f3){
                    //     if (e<e2){
                    //         int s = atomicSub(&support[e3], 1);
                    //         if (s == l){
                    //             n_mark[e3] = true;}
                    //     }
                    // }
                }
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

void peeling_direct_oi_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l){
    const int numRows = e_curr.size(0);
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l +=  1;
    peeling_edges_direct_oi<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        r_edges.data_ptr<int>(),
        re_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        numRows
    ); 
}

/*******************************************************peeling_direct_tile_oi*********************************************/
__global__ void peeling_edges_direct_tile_oi(
    const int *e_curr, const int *rows,
    const int *columns, const int *row_ptr, const int *r_edges, const int *re_ptr,  int *support, const bool *e_mask,
    bool *n_mark, const bool *in_curr, const int l, const int n_cut, const int numRows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRows) {
        int j = i / n_cut;
        int k = i  % n_cut;
        int e = e_curr[j];
        int row = rows[e];
        int col = columns[e];
        int idxA = row_ptr[row*n_cut+k];
        int endA = row_ptr[row*n_cut+k+1];
        int idxB = re_ptr[col*n_cut+k];
        int endB = re_ptr[col*n_cut+k+1];  
        while (idxA < endA && idxB < endB) {
            int colA = columns[idxA];
            int e3 = r_edges[idxB];
            int colB = rows[e3];
            if (colA == colB ) {
               //idxA是e1, e2<e1<e3
                if (e_mask[idxA] && e_mask[e3]){
                    bool f2 = in_curr[idxA];
                    bool f3 = in_curr[e3];
                    if (!(f2 || f3)){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;
                        }
                        s = atomicSub(&support[e3], 1);
                        if (s == l){
                            n_mark[e3] = true;
                        }   
                    }
                    else if(!f2 && f3){
                        int s = atomicSub(&support[idxA], 1);
                        if (s == l){
                            n_mark[idxA] = true;}
                    }
                    // else if(f2 && !f3){
                    //     if (e<e2){
                    //         int s = atomicSub(&support[e3], 1);
                    //         if (s == l){
                    //             n_mark[e3] = true;}
                    //     }
                    // }
                }
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

void peeling_direct_tile_oi_cuda(
    const torch::Tensor e_curr,
    const torch::Tensor rows,
    const torch::Tensor columns,
    const torch::Tensor row_ptr,
    const torch::Tensor r_edges,
    const torch::Tensor re_ptr,
    torch::Tensor support,
    const torch::Tensor e_mask, 
    torch::Tensor n_mark,
    const torch::Tensor in_curr, 
    const int l,
    const int n_cut){
    const int numRows = e_curr.size(0)*n_cut;
    const dim3 blocks((numRows + 512 - 1) / 512);
    const dim3 threads(512);
    // l +=  1;
    peeling_edges_direct_tile_oi<<<blocks, threads>>>(
        e_curr.data_ptr<int>(),
        rows.data_ptr<int>(),
        columns.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        r_edges.data_ptr<int>(),
        re_ptr.data_ptr<int>(),
        support.data_ptr<int>(),
        e_mask.data_ptr<bool>(),
        n_mark.data_ptr<bool>(),
        in_curr.data_ptr<bool>(),
        l,
        n_cut,
        numRows
    ); 
}