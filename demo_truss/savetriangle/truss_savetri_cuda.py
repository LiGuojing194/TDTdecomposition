import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import torch
import argparse
import time
sys.path.append('/home/zhangqi/workspace/TCRTruss32')
from src.type.Graph import Graph
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt
from torch_scatter import segment_csr, scatter
import os
from viztracer import VizTracer
"""
使用isin_cuda的存三角形的truss分解计算代码
truss_cuda_save->truss_cuda_save04->05
主要做两个改变，一个是mask = torch.isin(edges, e_curr) 、mask = torch.isin(e_affected, e_curr, assume_unique=True, invert=True)都用bukitize替换 (这个已经完成, 总共优化了0.3s)
另一个是segment_csr也用pycuda手写的求和函数替换，最后就可以换成所有计算都用int32存储啦（见truss_cuda_save05）
"""


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()

segment_code = """
    __global__ void segmentIsinKernel(
        const int *a_row_ptr, const int *a_col_indices,
        const int *b_mark,
        const int *b_row_ptr, const int *b_col_indices,
        int *output, const int numRows
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            int startA = a_row_ptr[row];
            int endA = a_row_ptr[row + 1];
            int startB = b_mark[row];
            int endB = b_row_ptr[startB + 1];
            startB = b_row_ptr[startB];

            for (int idxA = startA; idxA < endA; idxA++) {
                int colA = a_col_indices[idxA];
                // 二分查找 colA 在 B 中的位置
                int left = startB;
                int right = endB - 1;
                int mid = 0;
                while (left <= right) {
                    mid = left + (right - left) / 2;
                    if (b_col_indices[mid] == colA) {
                        output[idxA] = mid; 
                        //startB = mid + 1;
                        break;
                    } else if (b_col_indices[mid] < colA) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                startB = mid;
            }
        }
    }
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
    """

mod = SourceModule(segment_code)
segment_isin = mod.get_function("segmentIsinKernel")
segment_add = mod.get_function("segmentAddKernel")

def run_segment_isin(a_row_ptr, a_col_indices, b_mark, b_row_ptr, b_col_indices, output):
    numRows = np.int32(a_row_ptr.shape[0] - 1)
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_isin
    segment_isin(Holder(a_row_ptr), Holder(a_col_indices), Holder(b_mark), Holder(b_row_ptr), Holder(b_col_indices), Holder(output), numRows, block=block_size, grid=grid_size)

def run_segment_add(col_indices, col_ptr, output):
    numRows = np.int32(output.shape[0])
    block_size = (512, 1, 1)
    grid_size = (int(np.ceil(numRows / block_size[0])), 1, 1)
    global segment_add
    segment_add(Holder(col_indices), Holder(col_ptr), Holder(output), numRows, block=block_size, grid=grid_size)
        
########################
def intersection(values, boundaries): #value和mask都有序
    # mask = torch.full(values.shape, False, dtype=torch.bool, device='cuda')
    # v_end = torch.bucketize(boundaries[-1], values, right=True)
    # values = values[: v_end]
    mask = values<=boundaries[-1] #这个是顺序的，应该可以再次加速的
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result]==values
    return mask

def intersection_invert(values, boundaries):  #value和mask都有序
    mask = values <= boundaries[-1]
    values = values[mask]
    result = torch.bucketize(values, boundaries)
    mask[:result.shape[0]] = boundaries[result]==values
    return ~mask

def intersection_nosorted(values, boundaries): #value和mask都有序
    mask = values<=boundaries[-1]
    mask1 = torch.nonzero(mask).squeeze(1)
    values = values[mask1]
    result = torch.bucketize(values, boundaries)
    mask[mask1] = boundaries[result]==values
    return mask

def support_computing(graph: CSRCGraph):
    #对数据进行分块
    sizes_r = (graph.row_ptr[1:] - graph.row_ptr[:-1]) 
    ####################################根据每个顶点的邻居数量，进行分块处理如何快速分块，
    values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
    torch.cuda.synchronize()
    run_segment_add(sizes_r[graph.columns], graph.row_ptr, values)
    torch.cuda.synchronize()
    values = values.cumsum(0)
    batch = 200000000
    group = torch.searchsorted(values, torch.arange(0, values[-1]+batch-1, step=batch, dtype=torch.int64, device=graph.device), side = 'right')
    group[0] = 0
    group[-1] = values.shape[0]
    group = torch.unique(group)
    print("group shape", group.shape[0])
    del sizes_r, values
    torch.cuda.empty_cache()  #31ms
    # cuda_kernel = CudaKernel()
    ######################################
    torch.cuda.synchronize()
    t11 = time.time()
    support = torch.zeros(graph.columns.shape[0], dtype=torch.int32, device=graph.device) #1ms
    # edges = torch.tensor([], dtype=torch.int32, device=graph.device)  #160-167  运行一遍差不多1ms
    left_e = torch.tensor([], dtype=torch.int32, device=graph.device)
    right_e = torch.tensor([], dtype=torch.int32, device=graph.device)
    for start, end in zip(group[0:-1], group[1:]):
        #批量获取u邻居v的csr格式
        u_cs = graph.row_ptr[start]
        u_ce = graph.row_ptr[end]
        u_ptr = graph.row_ptr[start:end+1]-graph.row_ptr[start]
        #u开始标记
        u_r = torch.repeat_interleave(torch.arange(end-start, dtype=torch.int32, device=graph.device), u_ptr[1:] - u_ptr[:-1]) #1ms
        #批量获取v的邻居w
        v_c, v_ptr= batched_csr_selection_opt(graph.row_ptr[graph.columns[u_cs: u_ce]], graph.row_ptr[graph.columns[u_cs: u_ce]+1]) #调用batched_csr_selection_opt前的索引用了2.1ms，而这行代码用了17ms 很费时
        mask = torch.zeros(v_ptr[-1], dtype =torch.int32, device=graph.device) #5ms
        run_segment_isin(v_ptr, graph.columns[v_c], u_r, u_ptr,  graph.columns[u_cs: u_ce], mask)  #run_segment_isin前的索引4.6ms, isin运算少于23.5ms（和cat）
        #给三角形的三条边加上三角形数量  
        # e3
        mask1 = torch.nonzero(mask).squeeze(1)
        right_e = torch.cat([right_e, mask[mask1] + u_cs]) #Tensor.to 似乎用了至少十几ms
        # e2
        # mask = mask.to(torch.bool)
        left_e = torch.cat([left_e, v_c[mask1]])# mask.to(torch.bool)+v_c[mask] 7.2ms
        # e1
        values = torch.zeros(v_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
        mask[mask1] = torch.tensor(1, dtype=torch.int32, device=mask.device)
        run_segment_add(mask, v_ptr, values)
        support[u_cs: u_ce] += values #这行需要5.7ms  到再次循环的 torch.repeat_interleave 11ms
        # #e3
        # mask1 =  mask.to(torch.bool)
        # right_e = torch.cat([right_e, mask[mask.to(torch.bool)] + u_cs]) #Tensor.to 似乎用了至少十几ms
        # # e2
        # mask = mask.to(torch.bool)
        # left_e = torch.cat([left_e, v_c[mask]])# mask.to(torch.bool)+v_c[mask] 7.2ms
        # # e1
        # values = torch.zeros(v_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
        # run_segment_add(mask.to(torch.int32), v_ptr, values)
        # support[u_cs: u_ce] += values #这行需要5.7ms  到再次循环的 torch.repeat_interleave 11ms
    e_sizes = support.clone()   #浅拷贝，e_sizes的值会随着support的改变而改变 #24us
    unique_e, counts = torch.unique(left_e, return_counts=True)  #27 ms  #left的边号会更大，所以更费时间？
    support[unique_e] += counts
    unique_e, counts = torch.unique(right_e, return_counts=True)  #6ms
    support[unique_e] += counts
    torch.cuda.synchronize() #这到返回主函数0.2ms
    t22 = time.time()
    print('Support Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
    return support, left_e, right_e, e_sizes, t11

#这下面要编写不存三角形的truss分解代码
#先测试一下这个支持度计算编写的对不对？？？？
def truss_deposs(graph):
    #计算支持值  
    support, left_e, right_e, e_sizes, t11 = support_computing(graph)  
    # print("support:", support)
    # torch.cuda.empty_cache()
    #计算边映射序号             
    l = 1
    edges_id = torch.arange(graph.columns.shape[0], device=graph.device)
    mask = support.bool()
    e_rest = edges_id[mask]  #209-214 torch.repeat_interleave  0.3ms
    e_curr = e_rest[support[e_rest]==1]  #要是l=1时没有满足条件的边呢？？？
    while e_curr.shape[0] == 0:
        l +=1
        e_curr = e_rest[support[e_rest]==l] 
    edges = torch.repeat_interleave(e_rest, e_sizes[mask])  #0.22ms
    torch.cuda.empty_cache() #20ms
    # os.system('nvidia-smi')
    # return support+2, t11
    while True:
        mask = intersection(edges, e_curr)   #8ms
        # indece1_invert = torch.nonzero(~mask).squeeze()
        # ##########################################
        # mask = intersection_nosorted(right_e[indece1_invert], e_curr)
        # indece2_invert = torch.nonzero(~mask).squeeze()
        # ################################################
        # mask = intersection_nosorted(left_e[indece1_invert[indece2_invert]], e_curr) 
        # indece3_invert = torch.nonzero(~mask).squeeze()
        # indice_invert = indece1_invert[indece2_invert[indece3_invert]]
        # mask = torch.full(edges.shape, True, dtype=torch.bool, device=graph.device)
        # mask[indice_invert] = False
        # e_affected, a_counts = torch.unique(torch.cat([edges[mask], left_e[mask], right_e[mask]]), return_counts=True)
        # edges = edges[indice_invert]
        # left_e = left_e[indice_invert]
        # right_e = right_e[indice_invert]
        mask =  mask |  intersection_nosorted(right_e, e_curr) | intersection_nosorted(left_e, e_curr)  #20ms+15ms+  6.1ms(|| edges[mask], left_e[mask], right_e[mask])
        if torch.any(mask):
            m_indice = torch.nonzero(mask).squeeze(1) 
            e_affected, a_counts = torch.unique(torch.cat([edges[m_indice], left_e[m_indice], right_e[m_indice]]), return_counts=True)  #除去索引， 1.043ms
            m_indice = torch.nonzero(~mask).squeeze(1)  #13.5ms 222 这行到intersection_invert
            # print("m_indice shape", m_indice.shape)
            edges = edges[m_indice]
            left_e = left_e[m_indice]
            right_e = right_e[m_indice]
            mask = intersection_invert(e_affected, e_curr) #3ms
            e_affected = e_affected[mask] #这之后0.2ms
            support[e_affected] -= a_counts[mask]
            mask = support[e_affected] <= l
            support[e_affected[mask]] = l
            edges_id[e_curr] = -1
            e_curr = e_affected[mask]
        else:
            edges_id[e_curr] = -1
            e_curr.resize_(0)
        ########################################
        if e_curr.shape[0] == 0:
            if edges.shape[0] == 0:
                break
            #直接删除所有l层的边，可是得扫描一遍support，好慢
            # torch.cuda.empty_cache()
            e_rest = torch.unique(edges_id[e_rest])[1:] 
            l +=1
            e_curr = e_rest[support[e_rest]==l]
            while e_curr.shape[0] == 0:
                l +=1
                e_curr = e_rest[support[e_rest]==l] 
        # break
    return support+2, t11


def read_prepro_save(args):
    print('reading graph...', end=' ', flush=True) 
    graph, _= CSRGraph.read_graph(args.graph, directed=True)
    # graph, _= CSRGraph.read_graph(args.graph)
    torch.save(graph, args.output)
    print('Saving Done!')
    return None



def main_csrcgraph(args):
    print("------------------------------------------------")
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
   

    if args.cuda:
        graph.to('cuda')
        print('use cuda')

    # tracer = VizTracer()
    # tracer.start()
    support, t11 = truss_deposs(graph) 
    # tracer.stop()
    # tracer.save()
    torch.cuda.synchronize()
    t2 = time.time()
    print('Truss Completed! {}s time elapsed. Outputting results...'.format(t2 - t11))
    print('truss:{}'.format(support))
    sum = torch.max(support)
    print("k_max", sum)
    return t2-t11
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    # read_prepro_save(args)
    t_ls = []
    for i in range(2):
        tsec = main_csrcgraph(args)
        t_ls.append(tsec)
    print("mean time :", sum(t_ls)/len(t_ls))
    os.system('nvidia-smi')
    # a = torch.tensor([1, 1, 2, 0, 4, 2, 0, 3, 0])
    # indice = torch.argsort(a)
    # unique_e, counts_e = torch.unique_consecutive(a[indice], return_counts=True)
    # print(unique_e, inverse_indices)
    # b = torch.nonzero(a).squeeze()
    # print(b)
   




