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
是不是要在multigpu_truss_pre3的基础上改动，会比pre2跑的快一点？
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

###########################################################################
def k_truss(graph: CSRCOO, n_cut, num_v):  
    #e_mask标记剩余边
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=graph.device)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device = graph.device)
    #计算边映射序号             
    l = 1
    edges = torch.arange(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
    #第一步，整理整个图，支持度为零的数据清除
    e_mask = support.bool()
    edges = edges[e_mask]
    support = support[e_mask]
    graph.columns = graph.columns[e_mask]
    graph.rows = graph.rows[e_mask]
    graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
    e_curr = torch.where(support==l)[0].to(torch.int32)
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        e_curr = torch.where(support==l)[0].to(torch.int32)
    e_peeling_count = 0
    ###########e_mask = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    while True:
        # print("l:", l)
        while e_curr.shape[0] != 0: 
            e_truss = torch.cat([e_truss, edges[e_curr]])
            p = torch.unique(graph.rows[e_curr]) #这里面就不该有-1
            mask = torch.zeros(num_v, dtype =torch.bool, device=graph.device)
            mask[p] = True 
            mask = mask[graph.columns]  #python里索引最后一个就是-1
            p_c, _ = batched_csr_selection_opt2(graph.row_ptr[p*n_cut], graph.row_ptr[p*n_cut+n_cut])
            ###############mask[p_c] = ~e_mask[p_c]##############
            mask[p_c] = e_mask[p_c]
            e_affect = torch.where(mask)[0].to(torch.int32)
            ######################e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
            e_mask[e_curr] = False
            n_mark = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
            all_affect_support_not(e_affect, graph, n_cut, e_mask, l, n_mark, support)
            graph.columns[e_curr] = -1   #看看能不能把这行去掉
            e_peeling_count += e_curr.shape[0]
            if e_peeling_count > 10000000:  #  其实可以是每轮结束后判断要不要压缩
                # e_mask = ~e_mask
                ##############e_mask.logical_not_()
                support = support[e_mask]
                graph.columns = graph.columns[e_mask]
                graph.rows = graph.rows[e_mask]
                edges = edges[e_mask] 
                # values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
                # segment_add(e_mask.int(), graph.row_ptr, values)
                # graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0, dtype=torch.int32)])
                graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
                e_peeling_count = 0  
                e_curr = torch.where(support <= l)[0]  #####
                ################e_mask = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
                e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
            else:
                # e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
                e_curr = torch.where(n_mark)[0]
                n_mark = support[e_curr]<=l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
                e_curr = e_curr[n_mark]
            if (e_peeling_count + e_curr.shape[0]) == graph.columns.shape[0]:  #如何正确跳出循环
                e_truss = torch.cat([e_truss, edges[e_curr]])
                ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
                torch.cuda.synchronize()
                t22 = time.time()
                print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
                return l+2, t11, t22
        logging.info('{} level'.format(l))
        ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
        l += 1
        e_curr = torch.where(support == l)[0].to(torch.int32)  #也许这里
        while e_curr.shape[0] == 0:
            ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
            l += 1
            e_curr = torch.where(support == l)[0].to(torch.int32)
        if l > 20:
            break
    del n_mark, mask
    torch.cuda.synchronize()
    t4 = time.time()
    print('directed truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t4 - t11))
    logging.info("convert data  structure directed -> undirected")
    #增加张量edges_id_nbr, 更新row_ptr, 从而将有向图数据结构转成无向图数据结构
    #首先要压缩一遍图，剔除掉-1元素
    if not torch.all(e_mask):
        support = support[e_mask]
        graph.columns = graph.columns[e_mask]
        graph.rows = graph.rows[e_mask]
        edges = edges[e_mask]                  #这个没有考虑诶， 要不维护边号了，直接cat 行列边组
        graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
        e_peeling_count = 0  
        e_curr = torch.where(support <= l)[0].to(torch.int32)  
        e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    #更新row_ptr
    # print("graph.columns:", graph.columns, graph.columns.shape[0])
    r_edge = torch.argsort(graph.columns, stable=True).to(torch.int32)  #graph.columns里有-1怎么办？, 在这之前要压缩一次图
    # print("r_edge:", r_edge, r_edge.shape[0])
    if n_cut > 1: 
        tiling = (graph.row_ptr.shape[0]-1) // n_cut // n_cut
        print("tiling", tiling)
        tiling_block = graph.rows[r_edge]//(tiling+1) + graph.columns[r_edge]*n_cut
        print("tiling_block", tiling_block)
        e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True) #会有唯一元素-1，后面赋值的时候，会和最后一个元素重叠
        del tiling_block
    else: 
        e_u, e_counts = torch.unique_consecutive(graph.columns[r_edge], return_counts=True)
    size_r = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
    size_r[e_u] = e_counts.to(torch.int32)
    # print("count(size_r)", torch.sum(size_r))
    del e_u, e_counts
    # print("size_r:", size_r)
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
    # print("columns_g:", columns_g)
    # print("edges_id_nbr", edges_id_nbr)
    # print("row_ptr:", graph.row_ptr)
    torch.cuda.synchronize()
    t3 = time.time()
    #数据结构更新完之后，这里开始无向图上的分解
    in_curr = torch.zeros(support.shape[0], dtype=torch.bool, device= support.device)
    in_curr[e_curr] = True
    # edges_curr = torch.arange(in_curr.shape[0], dtype=torch.int32, device=device)
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
                print('undirected truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t3))
                return l+2, t11, t22
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
            #更新graph.row_ptr
            graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
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
        # e_curr = torch.arange(in_curr.shape[0], dtype=torch.int32, device=device)[in_curr]
        # if torch.sum(e_mask) < 20000:
        #     t22 = time.time()
        #     return l+2, t11, t22
        


           
            
    
    # torch.cuda.synchronize()
    # t22 = time.time()
    # print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    # return l+2, t11, t22

def csr_to_tilingcsr(graph: CSRCOO, tiling, n_cut):
    tiling_row_ptr = torch.zeros(graph.num_vertices*n_cut, dtype=torch.int32, device=graph.device)
    tiling_block = graph.columns//(tiling+1) + graph.rows*n_cut
    print("tiling_block", tiling_block)
    e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True)
    tiling_row_ptr[e_u] = e_counts.to(torch.int32)
    tiling_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), tiling_row_ptr.cumsum(0, dtype=torch.int32)])
    graph.row_ptr = tiling_row_ptr
    del tiling_row_ptr, tiling_block, e_u, e_counts
    # return tiling_row_ptr


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
    n_cut = 2
    num_v = graph.num_vertices
    new_csr_to_tilingcsr = torch.compile(csr_to_tilingcsr)
    if n_cut > 1:
        tiling = graph.num_vertices // n_cut
        # graph.row_ptr = csr_to_tilingcsr(graph, tiling, n_cut)
        new_csr_to_tilingcsr(graph, tiling, n_cut)
    # truss, t11, t22 = new_k_truss(graph, n_cut, num_v)
    truss, t11, t22 = k_truss(graph, n_cut, num_v)
    # print("e_rest row:", graph.rows[e_rest])
    # print("e_rest columns:", graph.columns[e_rest])
    # print("truss", truss)
    # print("max truss", torch.max(truss))
    # print('All triangle count Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args() 
    # read_prepro_save(args)
    for i in range(1):
        main_csrcgraph(args)

   