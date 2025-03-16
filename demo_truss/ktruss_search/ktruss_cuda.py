import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6, 7'
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import torch
import argparse
import time
sys.path.append('/root/autodl-tmp/TDTdecomposition')
from src.type.Graph import Graph
from src.type.CSRCOO import CSRCOO
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt


"""
#结果是正确的
ktruss_cuda2->ktruss_cuda2_v2
加上图压缩，当删除边标记数量超过剩下的一半等压缩一次图
#python  /home/zhangqi/workspace/TCRTruss32/src/test/ktruss_cuda.py  --graph '/home/zhangqi/workspace/data/cit-Patents-e.txt'  --output /home/zhangqi/workspace/output/citPatents_tri32.pth  --cuda
"""
# torch.cuda.set_device(0)
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

###########################################################################
def k_truss(graph: CSRCOO, l, n_cut, num_v):
    torch.cuda.synchronize()
    t11 = time.time()
    # flag = False
    support = support_computing_k(graph, n_cut, l)
    if l == 1:
        # support = torch.where(support<1, 2, 3)
        e_rest = torch.where(support>=1)[0]   #只要返回当前graph对应的编号就行
    # elif flag:  #对于小图就不压缩，整个一起重算
    #     mask = support >= l
    #     e_rest_len = support.shape[0] 
    #     e_rest = torch.where(mask)[0] 
    #     while e_rest.shape[0] < e_rest_len:
    #         e_rest_len = e_rest.shape[0]
    #         mask = support >= l
    #         e_rest = torch.where(mask)[0] 
    #         graph.columns[~mask] = -1
    #         support = support_computing_k_rest(graph, e_rest, n_cut, l)
    else:
        #计算边映射序号             
        mask = support < l  
        e_del = torch.where(mask)[0]
        ed_count = e_del.shape[0]
        while e_del.shape[0] !=0 and ed_count != support.shape[0]:  
            if ed_count >= (support.shape[0]/32):  #mask标记了所有要删除的边
            # if True:
                f_mask = ~mask
                e_rest = torch.where(f_mask)[0] 
                graph.columns = graph.columns[e_rest]
                graph.rows = graph.rows[e_rest]
                values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
                run_segment_add(f_mask.int().to(torch.int32), graph.row_ptr, values)
                graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0).to(torch.int32)])   #这里太混乱了，还是得修改读图函数，将graph.row和tiling_row_ptr合二为一
                support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut) #因为后面只对有向图中受影响边进行减，所以这里要计算出全部的支持值，不能只筛选了  #压缩图后不要重新计算支持度
                # support = support_computing_k(graph, n_cut, l)
                mask = support < l 
                e_del = torch.nonzero(mask).squeeze(1).to(torch.int32)  #要删除的边，support中对应的序号 
                ed_count = e_del.shape[0]
                # print("---------------------------compress------------------------")
                # print("e_del sum", torch.sum(mask))
                # print("support len: ", support.shape[0])
            else:
                #提取affected子图, 这个过程感觉有点复杂
                #子图受影响点
                point_affected = torch.unique(torch.cat([graph.rows[e_del], graph.columns[e_del]]))
                #使用一个顶点标记张量
                mask_v = torch.zeros(num_v+1, dtype =torch.bool, device=graph.device) 
                mask_v[point_affected] = True
                mask_a = mask_v[graph.columns]  #python里索引最后一个就是-1
                p_c, _ = batched_csr_selection_opt(graph.row_ptr[point_affected*n_cut], graph.row_ptr[point_affected*n_cut+n_cut])  #这里有问题n_cut>2时
                mask_a[p_c] = graph.columns[p_c] != -1  #或者  mask = mask_v[graph.rows] && mask_v[graph.columns]
                #受影响边的索引
                e_affect = torch.nonzero(mask_a).squeeze(1).to(torch.int32) 
                sub_rows = graph.rows[e_affect]
                sub_columns = graph.columns[e_affect]   #这里可以试试，用torch.nonzero(mask).squeeze(1)还是不用快
                sub_row_ptr = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
                run_segment_add(mask_a.int().to(torch.int32), graph.row_ptr, sub_row_ptr)
                sub_row_ptr = torch.cat([torch.zeros(1, dtype = torch.int32, device=graph.device), sub_row_ptr.cumsum(0).to(torch.int32)])
                sub_support = support[e_affect]   #这样的sub_support是新地址
                mark = mask[e_affect] #删除边标记, 还是torch.isin(e_affect, e_del)
                sub_affect_support(sub_rows, sub_columns, sub_row_ptr, n_cut, mark, sub_support)  #直接在原先的基础上减吧
                #找到受影响的非删除边
                # sub_support[mark] = 1000000
                # support[e_affect] = sub_support
                # graph.columns[e_del] = -1  #当前要删除的边, 这行操作有必要嘛
                # e_del = e_affect[sub_support < l]  #得剔除e_del吧  
                # mask[e_del] = True
                # ed_count += e_del.shape[0]
                support[e_affect] = sub_support
                s_mask = sub_support < l
                graph.columns[e_del] = -1  #当前要删除的边, 这行操作有必要嘛
                e_del = e_affect[s_mask & (~mark)]  #得剔除e_del吧  
                mask[e_del] = True
                ed_count += e_del.shape[0]
                # print("-------------------affected------------------")
                # print("e_del sum", torch.sum(mask))
                # print("support len: ", support.shape[0])
        e_rest = torch.nonzero(~mask).squeeze(1).to(torch.int32)
    torch.cuda.synchronize()
    t22 = time.time()
    return e_rest, t11, t22

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
    print("graph.rows shape", graph.rows.shape[0])
    print("graph.rows", graph.rows)
    print("graph.columns", graph.columns)
    print("graph.ptr", graph.row_ptr)
    # k = 3
    k = 36
    n_cut = 2
    num_v = graph.num_vertices
    if n_cut > 1:
        tiling = graph.num_vertices // n_cut
        graph.row_ptr = csr_to_tilingcsr(graph, tiling, n_cut)
    # print("support:", support)
    l = k-2
    e_rest, t11, t22 = k_truss(graph, l, n_cut, num_v)
    print("e_rest num", e_rest.shape[0])
    print("e_rest row:", graph.rows[e_rest])
    print("e_rest columns:", graph.columns[e_rest])
    print('All triangle count Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    # read_prepro_save(args)
    main_csrcgraph(args)
    # os.system('nvidia-smi')
   