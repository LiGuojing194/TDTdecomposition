import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import torch
import argparse
import time
sys.path.append('/root/autodl-tmp/TCRTruss32')
from src.type.Graph import Graph
from src.type.CSRCOO import CSRCOO
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt
# from torch_scatter import segment_csr, scatter


"""
ktruss_cuda2->ktruss_cuda2_v2->truss_cudan_n10_atomic.py->1 加上图压缩判断->2受影响子图一定要提取出来嘛
truss_cuda_10_atomic1->2  改成不提取受影响子图，并用e_truss和truss_ptr来存储分解结果
#python  /home/zhangqi/workspace/TCRTruss32/src/test/ktruss_cuda.py  --graph '/home/zhangqi/workspace/data/cit-Patents-e.txt'  --output /home/zhangqi/workspace/output/citPatents_tri32.pth  --cuda
"""
class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()

kernel_code = """
__global__ void segmentAddKernel(
        const int *col_indices,
        const int *col_ptr,
        int *output, const int numRows
    ) {
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
__global__ void checkElementsTwoPtr_k(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int k,
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
                if (output[idxA]<k){
                    atomicAdd(&output[idxA], 1);
                }
                if (output[idxB]<k){
                    atomicAdd(&output[idxB], 1);
                }
                count++;
                idxA++;
                idxB++;
            } else if (colA < colB) {
                idxA++;
            } else {
                idxB++;
            }
        }
        if (count>0 && output[row] < k){
            atomicAdd(&output[row], count);
        }
    }
}
__global__ void checkElementsTwoPtrTile_k(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut, const int k,
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
                if (output[startA]<k){
                    atomicAdd(&output[startA], 1);
                }
                if (output[startB]<k){
                    atomicAdd(&output[startB], 1);
                }
                count++;
                startA++;
                startB++;
            } else if (colA < colB) {
                startA++;
            } else {
                startB++;
            }
        }
        if (count>0 && output[i] < k){
                atomicAdd(&output[i], count);
            }
    }
}
__global__ void subAffectedSupport(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const bool *mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int col = b_col_indices[row];
        int startA = b_row_ptr[col];
        int endA = b_row_ptr[col+1];
        int a_row = a_rows[row];
        int startB = b_row_ptr[a_row];
        int endB = b_row_ptr[a_row+ 1];
        int count = 0;
        while (startA < endA && startB < endB) {
            int colA = b_col_indices[startA];
            int colB = b_col_indices[startB];
            if (colA == colB) {
                if (mark[row] || mark[startA] || mark[startB]){
                    if (!mark[startA]){
                    atomicSub(&output[startA], 1);
                    }
                    if (!mark[startB]){
                        atomicSub(&output[startB], 1);
                    }
                    if (!mark[row]){
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
                atomicSub(&output[row], count);
            }
    }
}
__global__ void subAffectedSupport_tile(
    const int *a_rows, const int *b_col_indices,
    const int *b_row_ptr, const int n_cut,  const bool *mark,
    int *output, const int numRows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        int i = row / n_cut;
        int j = row  % n_cut;
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
                if (mark[i] || mark[startA] || mark[startB]){
                    if (!mark[startA]){
                    atomicSub(&output[startA], 1);
                    }
                    if (!mark[startB]){
                        atomicSub(&output[startB], 1);
                    }
                    if (!mark[i]){
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
                atomicSub(&output[i], count);
            }
    }
}
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
"""
mod = SourceModule(kernel_code)
segment_add = mod.get_function("segmentAddKernel")
segment_isin_two = mod.get_function("checkElementsTwoPtr")
segment_isin_twotile = mod.get_function("checkElementsTwoPtrTile")
segment_isin_two_k = mod.get_function("checkElementsTwoPtr_k")
segment_isin_twotile_k = mod.get_function("checkElementsTwoPtrTile_k")
affected_support = mod.get_function("subAffectedSupport")
affected_support_tile = mod.get_function("subAffectedSupport_tile")
aaffected_support = mod.get_function("AllAffectedSupport")
aaffected_support_tile = mod.get_function("AllAffectedSupport_tile")
def run_segment_add(col_indices, col_ptr, output):
    numRows = np.int32(output.shape[0])
    block_size = (256, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_add
    segment_add(Holder(col_indices), Holder(col_ptr), Holder(output), numRows, block=block_size, grid=grid_size)
def run_kernel_two(a_rows, b_col_indices, b_row_ptr, output):
    numRows = output.shape[0]
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_isin_two
    segment_isin_two(Holder(a_rows), Holder(b_col_indices), Holder(b_row_ptr), Holder(output), np.int32(numRows),
    block=block_size, grid=grid_size)
def run_kernel_two_tile(a_rows, b_col_indices, b_row_ptr, n_cut, output):
    numRows = output.shape[0]*n_cut
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_isin_twotile
    segment_isin_twotile(Holder(a_rows), Holder(b_col_indices), Holder(b_row_ptr), np.int32(n_cut), Holder(output), np.int32(numRows),
    block=block_size, grid=grid_size)
def run_kernel_two_k(a_rows, b_col_indices, b_row_ptr, k, output):
    numRows = output.shape[0]
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_isin_two_k
    segment_isin_two_k(Holder(a_rows), Holder(b_col_indices), Holder(b_row_ptr), np.int32(k), Holder(output), np.int32(numRows),
    block=block_size, grid=grid_size)
def run_kernel_two_tile_k(a_rows, b_col_indices, b_row_ptr, n_cut, k, output):
    numRows = output.shape[0]*n_cut
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_isin_twotile_k
    segment_isin_twotile_k(Holder(a_rows), Holder(b_col_indices), Holder(b_row_ptr), np.int32(n_cut), np.int32(k), Holder(output), np.int32(numRows),
    block=block_size, grid=grid_size)
def run_affect_support(sub_rows, sub_columns, sub_row_ptr, mark, sub_support):
    numRows = sub_support.shape[0]
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global affected_support
    affected_support(Holder(sub_rows), Holder(sub_columns), Holder(sub_row_ptr), Holder(mark), Holder(sub_support), np.int32(numRows),
    block=block_size, grid=grid_size)  
def run_affect_support_tile(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support):
    numRows = sub_support.shape[0]*n_cut
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global affected_support_tile
    affected_support_tile(Holder(sub_rows), Holder(sub_columns), Holder(sub_row_ptr), np.int32(n_cut), Holder(mark), Holder(sub_support), np.int32(numRows),
    block=block_size, grid=grid_size)
#############################################
def run_aaffect_support(e_affect, a_rows, a_columns, a_row_ptr, mask, l, n_mark, support):
    numRows = e_affect.shape[0]
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global aaffected_support
    aaffected_support(Holder(e_affect), Holder(a_rows), Holder(a_columns), Holder(a_row_ptr), Holder(mask), np.int32(l), Holder(n_mark), Holder(support), np.int32(numRows),
    block=block_size, grid=grid_size)  
def run_aaffect_support_tile(e_affect,  a_rows, a_columns, a_row_ptr, n_cut, mask, l, n_mark, support):
    numRows = e_affect.shape[0]*n_cut
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global aaffected_support_tile
    aaffected_support_tile(Holder(e_affect), Holder(a_rows), Holder(a_columns), Holder(a_row_ptr), np.int32(n_cut), Holder(mask), np.int32(l), Holder(n_mark), Holder(support), np.int32(numRows),
    block=block_size, grid=grid_size)  

def intersection(values, boundaries): #value和mask都有序
    mask = values<=boundaries[-1] #这个是顺序的，应该可以再次加速的
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result]==values
    return mask

def intersection_nosorted(values, boundaries): #value和mask都有序
    mask = values<=boundaries[-1]
    mask1 = torch.nonzero(mask).squeeze(1)
    values = values[mask1]
    result = torch.bucketize(values, boundaries)
    mask[mask1] = boundaries[result]==values
    return mask

#####################################################################
# #分块支持度计算
# def support_computing_k(graph: CSRCOO, n_cut, k): #用于筛选出哪些边的支持度的值小于k
#     if n_cut > 1:
#         support_tile = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
#         run_kernel_two_tile_k(graph.rows, graph.columns, graph.row_ptr, n_cut, k, support_tile)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
#         return support_tile
#     else:
#         support = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
#         run_kernel_two_k(graph.rows, graph.columns, graph.row_ptr, k, support)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
#         return support
def support_computing(sub_rows, sub_colunms, tiling_row_ptr, n_cut): #用于计算出子图确切的支持度的值
    support_tile = torch.zeros(sub_colunms.shape[0], dtype=torch.int32, device=sub_colunms.device)
    if n_cut > 1:
        run_kernel_two_tile(sub_rows, sub_colunms, tiling_row_ptr, n_cut, support_tile)  
        return support_tile
    else:
        run_kernel_two(sub_rows, sub_colunms, tiling_row_ptr, support_tile) 
        return support_tile
#重新计算整个剩余边图中的支持度
# def support_computing_k_rest(graph: CSRCOO, e_rest, n_cut, l):
#     if n_cut > 1:
#         support_tile = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
#         run_kernel_two_tile_k(graph.rows, graph.columns, graph.row_ptr, n_cut, l, support_tile)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
#         return support_tile
#     else:
#         support = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
#         run_kernel_two_k(graph.rows, graph.columns, graph.row_ptr, l, support)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
#         return support

#子图求mark标记的边所拆除的三角形的对应边的支持度，，直接在sub_support上进行减
# def sub_affect_support(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support):
#     if n_cut>1:
#         run_affect_support_tile(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support)
#     else:
#         run_affect_support(sub_rows, sub_columns, sub_row_ptr, mark, sub_support)

#不提取子图的支持度减
#根据总图找到删除e_curr所拆除的三角形， 直接在support上减
def all_affect_support(e_affect, graph: CSRCOO, n_cut, mask,  l, n_mark, support):
    if n_cut>1:
        run_aaffect_support_tile(e_affect, graph.rows, graph.columns, graph.row_ptr, n_cut, mask,  l, n_mark, support)
    else:
        run_aaffect_support(e_affect, graph.rows, graph.columns, graph.row_ptr, mask,  l, n_mark, support)


###########################################################################
def k_truss(graph: CSRCOO, n_cut, num_v):
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=graph.device)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device = graph.device)
    #计算边映射序号             
    l = 1
    edges = torch.arange(graph.columns.shape[0], device=graph.device)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support.bool()
    support = support[mask]
    graph.columns = graph.columns[mask]
    graph.rows = graph.rows[mask]
    values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
    run_segment_add(mask.int().to(torch.int32), graph.row_ptr, values)
    graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0).to(torch.int32)])
    edges = edges[mask]
    e_curr = torch.where(support==l)[0]
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        e_curr = torch.where(support==l)[0]
    e_peeling_count = 0
    e_mask = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    while True:
        e_truss = torch.cat([e_truss, edges[e_curr]])
        p = torch.unique(graph.rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v, dtype =torch.bool, device=graph.device)
        mask_v[p] = True 
        mask = mask_v[graph.columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt(graph.row_ptr[p*n_cut], graph.row_ptr[p*n_cut+n_cut])
        # if (e_curr.shape[0]==1):
        #     print("p_c", p_c)
        #     print("e_curr", e_curr)
        #     print("e_mask[p_c]", e_mask[p_c])
        #     print("~e_mask[p_c]", ~e_mask[p_c])
        mask[p_c] = ~e_mask[p_c]
        #mask标记了需要查找三角形的边 从这里往下修改
        # e_affect = edges[mask]   #只需要对遍历这些边找到的三角形处理就行
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
        n_mark = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
        #必须传递一个标记删除边
        # print("l:", l)
        # print("e_affect:", e_affect.shape[0])
        all_affect_support(e_affect, graph, n_cut, e_mask, l, n_mark, support)
        graph.columns[e_curr] = -1   #看看能不能把这行去掉
        # support[e_curr] = l #增加一行矫正
        e_peeling_count += e_curr.shape[0]
        if e_peeling_count > 1000000:
            e_mask = ~e_mask
            support = support[e_mask]
            graph.columns = graph.columns[e_mask]
            graph.rows = graph.rows[e_mask]
            values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
            run_segment_add(e_mask.int().to(torch.int32), graph.row_ptr, values)
            graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0).to(torch.int32)])
            edges = edges[e_mask] 
            e_peeling_count = 0  
            # print("rest edges num:", edges.shape[0])
            e_curr = torch.where(support <= l)[0]
            # n_mark = n_mark[e_mask]
            # e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
            # n_mark = support[e_curr]<=l  
            # e_curr = e_curr[n_mark]
            e_mask = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
        else:
            e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
            n_mark = support[e_curr]<=l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
            e_curr = e_curr[n_mark]
        # print("e_curr:", e_curr.shape[0])
        # print("e_peeling_count:", e_peeling_count)
        if (e_peeling_count + e_curr.shape[0]) == graph.columns.shape[0]:  #如何正确跳出循环
            e_truss = torch.cat([e_truss, edges[e_curr]])
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
            break
        # support[e_curr] = l #增加一行矫正
        if e_curr.shape[0] == 0:
            print("before l:", l)
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
            l += 1
            e_curr = torch.where(support == l)[0]  #也许这里
            while e_curr.shape[0] == 0:
                ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
                l += 1
                # l = torch.min(support[~e_mask]).item()  #必须找到一个大于l的最小值的函数
                # print("l", l)
                e_curr = torch.where(support == l)[0] 
    torch.cuda.synchronize()
    t22 = time.time()
    return l+2, t11, t22

def csr_to_tilingcsr(graph: CSRCOO, tiling, n_cut):
    tiling_row_ptr = torch.zeros(graph.num_vertices*n_cut, dtype=torch.int32, device=graph.device)
    tiling_block = graph.columns//(tiling+1) + graph.rows*n_cut
    print("tiling_block", tiling_block)
    e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True)
    tiling_row_ptr[e_u] = e_counts.to(torch.int32)
    tiling_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), tiling_row_ptr.cumsum(0).to(torch.int32)])
    return tiling_row_ptr


def read_prepro_save(args):
    print('reading graph...', end=' ', flush=True) 
    graph, _= CSRCOO.read_graph(args.graph, directed=True)
    print(graph.row_ptr.dtype)
    torch.save(graph, args.output)
    print('Saving Done!')
    return None



def main_csrcgraph(args):
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
   

    if args.cuda:
        graph.to('cuda')
        print('use cuda')
    print("graph.rows", graph.rows)
    print("graph.columns", graph.columns)
    print("graph.ptr", graph.row_ptr)
    n_cut = 2
    num_v = graph.num_vertices
    if n_cut > 1:
        tiling = graph.num_vertices // n_cut
        graph.row_ptr = csr_to_tilingcsr(graph, tiling, n_cut)
    # print("support:", support)
    truss, t11, t22 = k_truss(graph, n_cut, num_v)
    # print("e_rest row:", graph.rows[e_rest])
    # print("e_rest columns:", graph.columns[e_rest])
    print("truss", truss)
    # print("max truss", torch.max(truss))
    print('All triangle count Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    # read_prepro_save(args)
    for i in range(2):
        main_csrcgraph(args)
    # os.system('nvidia-smi')
   