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
# import cuda_extension import segment_add, segment_isin2, segment_isin2tile, sub_AllAffectedSupport, sub_AllAffectedSupport_tile
# import segment_add_extension, segment_isin_extension
# from segment_add_extension import segment_add
# import mycudaf
from mycudaf import segment_isin2tile, sub_AllAffectedSupport, sub_AllAffectedSupport_tile
from mytensorf import segment_add, segment_isin2, segment_isin2tile

"""
ktruss_cuda2->ktruss_cuda2_v2->truss_cudan_n10_atomic.py->1 加上图压缩判断->2受影响子图一定要提取出来嘛
truss_cuda_10_atomic1->2  改成不提取受影响子图，并用e_truss和truss_ptr来存储分解结果
#python  /home/zhangqi/workspace/TCRTruss32/src/test/ktruss_cuda.py  --graph '/home/zhangqi/workspace/data/cit-Patents-e.txt'  --output /home/zhangqi/workspace/output/citPatents_tri32.pth  --cuda
"""
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
#################################################################################################
def support_computing(sub_rows, sub_colunms, tiling_row_ptr, n_cut): #用于计算出子图确切的支持度的值
    support_tile = torch.zeros(sub_colunms.shape[0], dtype=torch.int32, device=sub_colunms.device)
    if n_cut > 1:
        segment_isin2tile(sub_rows, sub_colunms, tiling_row_ptr, n_cut, support_tile)  
        return support_tile
    else:
        segment_isin2(sub_rows, sub_colunms, tiling_row_ptr, support_tile) 
        return support_tile
#不提取子图的支持度减
#根据总图找到删除e_curr所拆除的三角形， 直接在support上减
def all_affect_support(e_affect, graph: CSRCOO, n_cut, mask,  l, n_mark, support):
    if n_cut>1:
        sub_AllAffectedSupport_tile(e_affect, graph.rows, graph.columns, graph.row_ptr, n_cut, mask,  l, n_mark, support)
    else:
        sub_AllAffectedSupport(e_affect, graph.rows, graph.columns, graph.row_ptr, mask,  l, n_mark, support)


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
    segment_add(mask.int().to(torch.int32), graph.row_ptr, values)
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
            segment_add(e_mask.int().to(torch.int32), graph.row_ptr, values)
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--graph', type=str, help='path to graph', required=True)
    # parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    # parser.add_argument('--cuda', action='store_true', help='use cuda')
    # args = parser.parse_args() 
    # # read_prepro_save(args)
    # for i in range(2):
    #     main_csrcgraph(args)
    # os.system('nvidia-smi')

    # a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device='cuda')
    # b = torch.tensor([0, 3, 5, 8], dtype=torch.int32, device='cuda')
    # c = torch.tensor([0, 0, 0], dtype=torch.int32, device='cuda')
    # torch.cuda.synchronize()
    # t11 = time.time()
    # for i in range(3):
    #     segment_add(a,b,c)
    # torch.cuda.synchronize()
    # t22 = time.time()
    # print("c:", c)
    # print(' {}s time elapsed on gpu. Outputting results...'.format(t22 - t11))
    # aa = torch.tensor([1, 2, 3, 4, 5, 6, 7, 9], dtype=torch.int32)
    # bb = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    # cc = torch.tensor([0, 0, 0], dtype=torch.int32)
    # torch.cuda.synchronize()
    # t11 = time.time()
    # for i in range(3):
    #     segment_add(aa, bb, cc)
    # torch.cuda.synchronize()
    # t22 = time.time()
    # print(' {}s time elapsed on cpu. Outputting results...'.format(t22 - t11))
    # print("ccpu:", cc)
    print("#########################################################")
    # s_row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3], dtype=torch.int32)
    # s_columns = torch.tensor([1, 4, 2, 3, 4, 3, 4, 4], dtype=torch.int32)
    # s_row_ptr = torch.tensor([0, 2, 5, 7, 8], dtype=torch.int32)
    # output = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    # torch.cuda.synchronize()
    # t11 = time.time()
    # for i in range(3):
    #     segment_isin2(s_row, s_columns, s_row_ptr, output)
    # torch.cuda.synchronize()
    # t22 = time.time()
    # print(' {}s time elapsed on cpu. Outputting results...'.format(t22 - t11))
    # print("cpu support output:", output)
    # s_row = s_row.pin_memory().to('cuda')
    # s_columns = s_columns.pin_memory().to('cuda')
    # s_row_ptr = s_row_ptr.pin_memory().to('cuda')
    # output = output.pin_memory().to('cuda')
    # torch.cuda.synchronize()
    # t11 = time.time()
    # for i in range(3):
    #     segment_isin2(s_row, s_columns, s_row_ptr, output)
    # torch.cuda.synchronize()
    # t22 = time.time()
    # print(' {}s time elapsed on gpu. Outputting results...'.format(t22 - t11))
    # print("gpu support output:", output)
    ############################################################################
    s_row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3], dtype=torch.int32)
    s_columns = torch.tensor([1, 4, 2, 3, 4, 3, 4, 4], dtype=torch.int32)
    s_row_ptr = torch.tensor([0, 1, 2, 2, 5, 5, 7, 7, 8], dtype=torch.int32)
    output = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    n_cut = 2
    torch.cuda.synchronize()
    t11 = time.time()
    for i in range(3):
        segment_isin2tile(s_row, s_columns, s_row_ptr, n_cut, output)
    torch.cuda.synchronize()
    t22 = time.time()
    print(' {}s time elapsed on cpu. Outputting results...'.format(t22 - t11))
    print("cpu support output:", output)
    s_row = s_row.pin_memory().to('cuda')
    s_columns = s_columns.pin_memory().to('cuda')
    s_row_ptr = s_row_ptr.pin_memory().to('cuda')
    output = output.pin_memory().to('cuda')
    torch.cuda.synchronize()
    t11 = time.time()
    for i in range(3):
        segment_isin2tile(s_row, s_columns, s_row_ptr, n_cut, output)
    torch.cuda.synchronize()
    t22 = time.time()
    print(' {}s time elapsed on gpu. Outputting results...'.format(t22 - t11))
    print("gpu support output:", output)
    # print(segment_add_extension.__file__)

   