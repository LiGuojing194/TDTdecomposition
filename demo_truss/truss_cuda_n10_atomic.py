import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
import numpy as np
import sys
import torch
import argparse
import time
import pycuda.autoinit
from pycuda.compiler import SourceModule
# print("start")
sys.path.append('/root/autodl-tmp/TCRTruss32')
from src.type.CSRCOO import CSRCOO
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt
# from torch_scatter import segment_csr, scatter
# print("start")


"""
ktruss_cuda2->ktruss_cuda2_v2->truss_cudan_n10_atomic.py
从ktruss查询改成truss全分解
加上图压缩，当删除边标记数量超过剩下的一半等压缩一次图
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
"""
mod = SourceModule(kernel_code)
segment_add = mod.get_function("segmentAddKernel")
segment_isin_two = mod.get_function("checkElementsTwoPtr")
segment_isin_twotile = mod.get_function("checkElementsTwoPtrTile")
segment_isin_two_k = mod.get_function("checkElementsTwoPtr_k")
segment_isin_twotile_k = mod.get_function("checkElementsTwoPtrTile_k")
affected_support = mod.get_function("subAffectedSupport")
affected_support_tile = mod.get_function("subAffectedSupport_tile")
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

def intersection(values, boundaries): #value和mask都有序
    mask = values<=boundaries[-1] #这个是顺序的，应该可以再次加速的
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result]==values
    return mask

#####################################################################
#分块支持度计算
def support_computing_k(graph: CSRCOO, n_cut, k): #用于筛选出哪些边的支持度的值小于k
    if n_cut > 1:
        support_tile = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
        run_kernel_two_tile_k(graph.rows, graph.columns, graph.row_ptr, n_cut, k, support_tile)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
        return support_tile
    else:
        support = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
        run_kernel_two_k(graph.rows, graph.columns, graph.row_ptr, k, support)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
        return support
def support_computing(sub_rows, sub_colunms, tiling_row_ptr, n_cut): #用于计算出子图确切的支持度的值
    support_tile = torch.zeros(sub_colunms.shape[0], dtype=torch.int32, device=sub_colunms.device)
    if n_cut > 1:
        run_kernel_two_tile(sub_rows, sub_colunms, tiling_row_ptr, n_cut, support_tile)  
        return support_tile
    else:
        run_kernel_two(sub_rows, sub_colunms, tiling_row_ptr, support_tile) 
        return support_tile
#重新计算整个剩余边图中的支持度
def support_computing_k_rest(graph: CSRCOO, e_rest, n_cut, l):
    if n_cut > 1:
        support_tile = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
        run_kernel_two_tile_k(graph.rows, graph.columns, graph.row_ptr, n_cut, l, support_tile)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
        return support_tile
    else:
        support = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device)
        run_kernel_two_k(graph.rows, graph.columns, graph.row_ptr, l, support)  #*************************如果直接放到显存里硬算法会怎样#避免原子操作，应该可以加速计算吧
        return support

#子图求mark标记的边所拆除的三角形的对应边的支持度，，直接在sub_support上进行减
def sub_affect_support(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support):
    if n_cut>1:
        run_affect_support_tile(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support)
    else:
        run_affect_support(sub_rows, sub_columns, sub_row_ptr, mark, sub_support)


# flag = False
# elif flag:  #对于小图就不压缩，整个一起重算   #这个功能后面添加吧
#     mask = support >= l
#     e_rest_len = support.shape[0] 
#     e_rest = torch.where(mask)[0] 
#     while e_rest.shape[0] < e_rest_len:
#         e_rest_len = e_rest.shape[0]
#         mask = support >= l
#         e_rest = torch.where(mask)[0] 
#         graph.columns[~mask] = -1
#         support = support_computing_k_rest(graph, e_rest, n_cut, l)
###########################################################################
def k_truss(graph: CSRCOO, n_cut, num_v):
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)
    #计算边映射序号             
    l = 1
    edges = torch.arange(graph.columns.shape[0], device=graph.device)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support.bool()
    graph.columns = graph.columns[mask]
    graph.rows = graph.rows[mask]
    values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
    run_segment_add(mask.int().to(torch.int32), graph.row_ptr, values)
    graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0).to(torch.int32)])
    edges = edges[mask]
    e_curr = torch.where(support[edges]==l)[0]
    while e_curr.shape[0] == 0:
        l += 1
        e_curr = torch.where(support[edges]==l)[0] 
    while True:
        p = torch.unique(torch.cat([graph.rows[e_curr], graph.columns[e_curr]]))
        mask_v = torch.zeros(num_v+1, dtype =torch.bool, device=graph.device)
        mask_v[p] = True
        mask = mask_v[graph.columns]  #python里索引最后一个就是-1
        # mask = mask_v[graph.columns] | mask_v[graph.rows]  #mask_v[graph.rows]还得不等于-1
        p_c, _ = batched_csr_selection_opt(graph.row_ptr[p*n_cut], graph.row_ptr[p*n_cut+n_cut])
        mask[p_c] = graph.columns[p_c] != -1
        sub_rows = graph.rows[mask]
        sub_columns = graph.columns[mask]   #这里可以试试，用torch.nonzero(mask).squeeze(1)还是不用快
        sub_row_ptr = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
        run_segment_add(mask.int().to(torch.int32), graph.row_ptr, sub_row_ptr)
        sub_row_ptr = torch.cat([torch.zeros(1, dtype = torch.int32, device=graph.device), sub_row_ptr.cumsum(0).to(torch.int32)])
        sub_edges = torch.arange(graph.columns.shape[0], dtype = torch.int32, device=graph.device)[mask]
        #接下来找到子图中要拆除的所有的三角形
        e_affect = edges[sub_edges]
        sub_support = support[e_affect]   #这样的sub_support是新地址
        # mark = mask[sub_edges] #这个要怎么标记呀，呜呜###################################################################
        # mark = intersection(sub_edges, e_curr)
        mark = sub_support == l
        sub_affect_support(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support)  #直接在原先的基础上减吧
        support[e_affect] = sub_support
        mask = sub_support < l
        support[e_affect[mask]] = l
        graph.columns[e_curr] = -1
        mask = sub_support <= l
        e_curr = sub_edges[mask& (~mark)]  
        if e_curr.shape[0] == 0:
            #进行一次子图计算, 先更新columns
            mask = graph.columns != -1
            graph.columns = graph.columns[mask]
            if graph.columns.shape[0] == 0:
                break
            #更新graph.row_ptr
            graph.rows = graph.rows[mask]
            values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
            run_segment_add(mask.int().to(torch.int32), graph.row_ptr, values)
            graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0).to(torch.int32)])
            edges = edges[mask] 
            l += 1
            e_curr = torch.where(support[edges] == l)[0]
            while e_curr.shape[0] == 0:
                l += 1
                e_curr = torch.where(support[edges] == l)[0] 
    torch.cuda.synchronize()
    t22 = time.time()
    return support+2, t11, t22

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
    # k=3
    # k = 159
    n_cut = 1
    num_v = graph.num_vertices
    if n_cut > 1:
        tiling = graph.num_vertices // n_cut
        graph.row_ptr = csr_to_tilingcsr(graph, tiling, n_cut)
    # print("support:", support)
    truss, t11, t22 = k_truss(graph, n_cut, num_v)
    # print("e_rest row:", graph.rows[e_rest])
    # print("e_rest columns:", graph.columns[e_rest])
    print("truss", truss)
    print("max truss", torch.max(truss))
    print('All triangle count Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    read_prepro_save(args)
    main_csrcgraph(args)
    # os.system('nvidia-smi')
   