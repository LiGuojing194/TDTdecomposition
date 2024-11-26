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
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt2, batched_csr_selection_opt
from mytensorf import segment_add, segment_isin2, segment_isin2tile, sub_AllAffectedSupport, sub_AllAffectedSupport_tile, sub_AllAffectedSupport_not, sub_AllAffectedSupport_tilenot, peeling_undirect_tile, peeling_undirect
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

"""
修改代码，将代码修改成边数每减少1/2， 分块数就乘以2
在multigpu_truss_pre2_undirect.py的基础上，实现的用了cuda的纯无向图的truss分解。
除了分块以外，数据结构和已有的无向图上的truss分解代码，大致相同
python /root/autodl-tmp/TCRTruss32/src/demo_truss/multigpu_truss.py  --graph /root/autodl-tmp/data/com-orkut.ungraph.txt  --output /root/autodl-tmp/output/com_orkut_truss.pth  --cuda
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

def all_affect_support_not(e_affect, graph: CSRCOO, n_cut, mask,  l, n_mark, support):
    if n_cut>1:
        sub_AllAffectedSupport_tilenot(e_affect, graph.rows, graph.columns, graph.row_ptr, n_cut, mask,  l, n_mark, support)
    else:
        sub_AllAffectedSupport_not(e_affect, graph.rows, graph.columns, graph.row_ptr, mask,  l, n_mark, support)

def peeling_edges_undirect(e_curr, rows, columns,  columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l, n_cut):
    if n_cut>1:
        peeling_undirect_tile(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l, n_cut)
    else:
        peeling_undirect(e_curr, rows, columns, columns_g, row_ptr, edges_id_nbr, support, e_mask, n_mark, in_curr, l)
###########################################################################
def update_row_ptr(e_mask, row_ptr):
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    segment_add(e_mask.int(), row_ptr, values)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), values.cumsum(0, dtype=torch.int32)])
    return row_ptr

def update_row_ptr2(e_mask, row_ptr):
    # values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    # segment_add(e_mask.int(), row_ptr, values)
    # row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), values.cumsum(0, dtype=torch.int32)])
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), torch.cumsum(e_mask.int(), 0, dtype=torch.int32)])[row_ptr]
###########################################################################
def preprocess(graph: CSRCOO, n_cut_s, n_cut):  
    num_v = graph.num_vertices
    new_csr_to_tilingcsr = torch.compile(csr_to_tilingcsr)
    if n_cut_s > 1:
        tiling = num_v // n_cut_s
        directed_ptr = new_csr_to_tilingcsr(graph, tiling, n_cut_s)
    else:
        directed_ptr = graph.row_ptr
    #更新row_ptr
    r_edge = torch.argsort(graph.columns, stable=True).to(torch.int32)  #graph.columns里有-1怎么办？, 在这之前要压缩一次图
    if n_cut > 1: 
        tiling = num_v // n_cut
        tiling_block = graph.rows[r_edge]//(tiling+1) + graph.columns[r_edge]*n_cut
        # print("tiling_block", tiling_block)
        e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True) #会有唯一元素-1，后面赋值的时候，会和最后一个元素重叠
        del tiling_block
        graph.row_ptr = new_csr_to_tilingcsr(graph, tiling, n_cut)
    else: 
        e_u, e_counts = torch.unique_consecutive(graph.columns[r_edge], return_counts=True)
    size_r = torch.zeros(num_v*n_cut, dtype=torch.int32, device=graph.device)
    size_r[e_u] = e_counts.to(torch.int32)
    del e_u, e_counts
    graph.row_ptr[1:] += size_r.cumsum(0, dtype=torch.int32)   
    #生成edges_id_nbr 和 columns_g
    edges_id_nbr= torch.zeros(graph.columns.shape[0]*2, dtype=torch.int32, device=graph.row_ptr.device)
    columns_g = torch.zeros(graph.columns.shape[0]*2, dtype=torch.int32, device=graph.row_ptr.device)
    indice, _ = batched_csr_selection_opt(graph.row_ptr[:-1]+size_r, graph.row_ptr[1:])
    # print("indice: ", indice)
    edges_id_nbr[indice] = torch.arange(graph.columns.shape[0], dtype=torch.int32, device=graph.row_ptr.device)
    columns_g[indice] = graph.columns
    # print("graph.row_ptr[:-1]:", graph.row_ptr[:-1])
    # print("graph.row_ptr[:-1]+size_r:", graph.row_ptr[:-1]+size_r)
    indice, _ = batched_csr_selection_opt(graph.row_ptr[:-1], graph.row_ptr[:-1]+size_r)
    # print("indice: ", indice)
    edges_id_nbr[indice] = r_edge
    columns_g[indice] = graph.rows[r_edge]
    del r_edge, size_r, indice
    # print("directed_ptr:", directed_ptr)
    # print("columns_g: ",columns_g)
    # print("edges_id_nbr: ", edges_id_nbr)
    # print("graph.row_ptr:", graph.row_ptr)
    return graph, directed_ptr, columns_g, edges_id_nbr



def undirected_truss(graph: CSRCOO, columns_g, edges_id_nbr, directed_ptr, n_cut_s, n_cut, num_v):
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    support = support_computing(graph.rows, graph.columns, directed_ptr, n_cut_s)
    del directed_ptr
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=graph.device)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device=graph.device)
    #计算边映射序号             
    l = 1
    edges = torch.arange(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
    #第一步，整理整个图，支持度为零的数据清除
    e_mask = support.bool()
    # # e_peeling_count = graph.columns.shape[0] - torch.sum(e_mask)
    # # if e_peeling_count > 10000000:
    #     print("compress")
    support = support[e_mask]
    graph.columns = graph.columns[e_mask]
    graph.rows = graph.rows[e_mask]
    edges = edges[e_mask] 
    #更新edges_id_nbr
    e_map = torch.cat([torch.zeros(1, dtype=torch.int32, device=e_mask.device), torch.cumsum(e_mask.int(), 0, dtype=torch.int32)])   #or e_map = torch.zeros  e_map[e_mask] = torch.arrange
    e_mask = e_mask[edges_id_nbr]
    # print("columns_g shape:", columns_g.shape[0])
    # print("e_mask shape:", e_mask.shape[0])
    columns_g = columns_g[e_mask]
    edges_id_nbr = edges_id_nbr[e_mask]  
    edges_id_nbr = e_map[edges_id_nbr]
    del e_map
    #更新graph.row_ptr
    graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
    ###########################
    e_peeling_count = 0  
    # e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    ##########################################
    # edges = edges[e_mask]
    # support = support[e_mask]
    # graph.columns = graph.columns[e_mask]
    # graph.rows = graph.rows[e_mask]
    # graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
    in_curr = support==l
    while not torch.any(in_curr):
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        in_curr = support==l
    e_curr = torch.where(in_curr)[0].to(torch.int32)
    ############################################################################
    # in_curr = torch.zeros(support.shape[0], dtype=torch.bool, device= support.device)
    # in_curr[e_curr] = True
    e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    col_count = graph.columns.shape[0]
    while True:
        while e_curr.shape[0] != 0:
            # print("e_curr:", e_curr)
            e_truss = torch.cat([e_truss, edges[e_curr]])
            n_mark = torch.zeros(support.shape[0], dtype=torch.bool, device= support.device)
            # print("e_mask:", e_mask)
            # peeling_edges_undirect(e_curr, graph.rows, graph.columns,  graph.row_ptr, edges_id_nbr, support, e_mask, n_mark, l, n_cut)#####################这个函数需要修改
            # print(e_curr.dtype, graph.rows.dtype, graph.columns.dtype, graph.row_ptr.dtype, edges_id_nbr.dtype, support.dtype, e_mask.dtype, n_mark.dtype)
            # print("suppport:", support)
            # in_curr = torch.(support.shape[0], dtype=torch.bool, device= support.device)
            # in_curr[e_curr] = True
            peeling_edges_undirect(e_curr, graph.rows, graph.columns, columns_g, graph.row_ptr, edges_id_nbr, support, e_mask, n_mark,in_curr, l, n_cut)
            # print("suppport:", support)
            e_mask[e_curr] = False
            e_peeling_count += e_curr.shape[0]
            in_curr = n_mark
            # e_curr = edges_curr[]
            e_curr = torch.where(n_mark)[0].int()
            # print("e_peeling_count + e_curr.shape[0]", e_peeling_count + e_curr.shape[0])
            # print("graph.columns.shape[0]:", graph.columns.shape[0])
            # print("e_mask sum:", torch.sum(e_mask))
            if (e_peeling_count + e_curr.shape[0]) == graph.columns.shape[0]:  #如何正确跳出循环
                # print("break")
                e_truss = torch.cat([e_truss, edges[e_curr]])
                ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
                torch.cuda.synchronize()
                t22 = time.time()
                print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
                return l+2
        if e_peeling_count > 10000000:
            support = support[e_mask]
            graph.columns = graph.columns[e_mask]
            graph.rows = graph.rows[e_mask]
            edges = edges[e_mask] 
            #更新edges_id_nbr
            e_map = torch.cat([torch.zeros(1, dtype=torch.int32, device=e_mask.device), torch.cumsum(e_mask.int(), 0, dtype=torch.int32)])   #or e_map = torch.zeros  e_map[e_mask] = torch.arrange
            e_mask = e_mask[edges_id_nbr]
            # print("columns_g shape:", columns_g.shape[0])
            # print("e_mask shape:", e_mask.shape[0])
            columns_g = columns_g[e_mask]
            edges_id_nbr = edges_id_nbr[e_mask]  
            edges_id_nbr = e_map[edges_id_nbr]
            del e_map
            graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
            #更新graph.row_ptr (是相对于columns_g的指针)
            if (col_count/graph.columns.shape[0])>2 and n_cut < 64:
                print("Add n_cut:", n_cut)
                col_count = graph.columns.shape[0]
                # print("graph.row_ptr: ", graph.row_ptr)
                temp_ptr = graph.row_ptr[::n_cut]
                # print("temp_ptr: ", temp_ptr)
                rows_g = torch.repeat_interleave(torch.arange(num_v, dtype=torch.int32, device=graph.device), temp_ptr[1:]-temp_ptr[:-1])  #后续可以写一个pytorch算子来优化这个repeat_interleave算子
                n_cut = n_cut*2
                tiling = num_v // n_cut
                tiling_row_ptr = torch.zeros(num_v*n_cut, dtype=torch.int32, device=graph.device)
                tiling_block = columns_g//(tiling+1) + rows_g*n_cut
                # print("tiling_block", tiling_block)
                e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True)
                tiling_row_ptr[e_u] = e_counts.to(torch.int32)
                graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), tiling_row_ptr.cumsum(0, dtype=torch.int32)])
                del tiling_row_ptr, tiling_block, e_u, e_counts, rows_g, temp_ptr
                # break
            # else:
                # graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
            ###########################
            e_peeling_count = 0  
            e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
        logging.info('{} level'.format(l))
        ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
        l += 1
        # e_curr = torch.where(support == l)[0].to(torch.int32)
        in_curr = support == l
        # while e_curr.shape[0] == 0:
        while not torch.any(in_curr):
            ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
            l += 1
            in_curr = support == l
            # e_curr = torch.where(support == l)[0].to(torch.int32)
        e_curr = torch.where(in_curr)[0].to(torch.int32)

        

def csr_to_tilingcsr(graph: CSRCOO, tiling, n_cut):
    tiling_row_ptr = torch.zeros(graph.num_vertices*n_cut, dtype=torch.int32, device=graph.device)
    tiling_block = graph.columns//(tiling+1) + graph.rows*n_cut
    print("tiling_block", tiling_block)
    e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True)
    tiling_row_ptr[e_u] = e_counts.to(torch.int32)
    tiling_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), tiling_row_ptr.cumsum(0, dtype=torch.int32)])
    # graph.row_ptr = tiling_row_ptr
    del tiling_block, e_u, e_counts
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
    graph.row_ptr = graph.row_ptr.to(torch.int32)
    graph.pin_memory()
   

    if args.cuda:
        graph.to('cuda')
        print('use cuda')
    print("graph.rows shape:", graph.rows.shape[0])
    print("graph.columns shape:", graph.columns.shape[0])
    print("graph.ptr shape:", graph.row_ptr.shape[0])
    n_cut_s = 2
    n_cut = 1
    num_v = graph.num_vertices
    graph, directed_ptr, columns_g, edges_id_nbr = preprocess(graph, n_cut_s, n_cut)
    max_t = undirected_truss(graph, columns_g, edges_id_nbr, directed_ptr, n_cut_s, n_cut, num_v)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args() 
    read_prepro_save(args)
    # for i in range(1):
    #     main_csrcgraph(args)

   