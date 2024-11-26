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
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt2
from mytensorf import segment_add, segment_isin2, segment_isin2tile, sub_AllAffectedSupport, sub_AllAffectedSupport_tile, sub_AllAffectedSupport_not, sub_AllAffectedSupport_tilenot, sub_AllAffectedSupport_notwin, sub_AllAffectedSupport_tilenotwin
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
"""
#2.用inwindow标记在索引范围内的边
#3.每层结束后再压缩图
尝试torch.complier, 失败了
ktruss_cuda2->ktruss_cuda2_v2->truss_cudan_n10_atomic.py->1 加上图压缩判断->2受影响子图一定要提取出来嘛
truss_cuda_10_atomic1->2  改成不提取受影响子图，并用e_truss和truss_ptr来存储分解结果
先写一个函数找到分割的l值，再写一个函数可以从l开始分解
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
def all_affect_support_not_win(e_affect, graph: CSRCOO, n_cut, mask,  l, n_mark, support, windows_high, in_windows_mask, buff_index, buf_size):
    if n_cut>1:
        sub_AllAffectedSupport_tilenotwin(e_affect, graph.rows, graph.columns, graph.row_ptr, n_cut, mask,  l, n_mark, support, windows_high, in_windows_mask, buff_index, buf_size)
    else:
        sub_AllAffectedSupport_not_win(e_affect, graph.rows, graph.columns, graph.row_ptr, mask,  l, n_mark, support, windows_high, in_windows_mask, buff_index, buf_size)
###########################################################################
def update_row_ptr(e_mask, row_ptr):
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    segment_add(e_mask.int(), row_ptr, values)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), values.cumsum(0, dtype=torch.int32)])
    return row_ptr

###########################################################################
def k_truss(graph: CSRCOO, n_cut, num_v):  
    #e_mask标记剩余边，
    #
    #2.用inwindow标记在索引范围内的边
    #3.每层结束后再压缩图
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
    graph.columns = graph.columns[edges]
    graph.rows = graph.rows[edges]
    graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
    edges_curr = torch.arange(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
    buff_index = torch.zeros(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
    buf_size = torch.zeros(1, dtype= torch.int32, device=graph.device)
    # e_curr = torch.where(support==l)[0]
    #初始化in_windows_mask 和 support_buff_index
    windows_skip = 16
    windows_low = l
    windows_high = l+windows_skip
    in_windows_mask = support < windows_high 
    buff = edges_curr[in_windows_mask]
    buf_size[0] = buff.shape[0]
    # buff_index[0] = buf_size + 1
    buff_index[:buf_size[0]] = buff
    e_curr = buff[support[buff]==l]
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        if l == windows_high:
            windows_high = l+windows_skip
            in_windows_mask = (support >= l) & (support< windows_high)
            buff = edges_curr[in_windows_mask]
            buf_size[0] = buff.shape[0]
            buff_index[:buf_size[0]] = buff
        e_curr = buff[support[buff]==l]
    e_peeling_count = 0
    e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
    while True:
        # print("l:", l)
        # print("e_curr", e_curr)
        # print("n_cut", n_cut)
        # print("buffer_size", buf_size[0])
        # break
        e_truss = torch.cat([e_truss, edges[e_curr]])
        p = torch.unique(graph.rows[e_curr]) #这里面就不该有-1
        mask = torch.zeros(num_v, dtype =torch.bool, device=graph.device)
        mask[p] = True 
        mask = mask[graph.columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(graph.row_ptr[p*n_cut], graph.row_ptr[p*n_cut+n_cut])
        ###############mask[p_c] = ~e_mask[p_c]##############
        mask[p_c] = e_mask[p_c]
        # mask[p_c] = e_mask[p_c].logical_not_()
        #mask标记了需要查找三角形的边 从这里往下修改
        # e_affect = edges[mask]   #只需要对遍历这些边找到的三角形处理就行
        # e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        # e_affect = torch.where(mask)[0].to(torch.int32)
        e_affect = edges_curr[mask]
        ######################e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
        e_mask[e_curr] = False
        n_mark = torch.zeros(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
        # print("in_windows_mask", in_windows_mask)
        all_affect_support_not_win(e_affect, graph, n_cut, e_mask, l, n_mark, support, windows_high, in_windows_mask, buff_index, buf_size)
        torch.cuda.synchronize()
        graph.columns[e_curr] = -1   #看看能不能把这行去掉
        e_peeling_count += e_curr.shape[0]
        if e_peeling_count > 10000000:
            support = support[e_mask]
            graph.columns = graph.columns[e_mask]
            graph.rows = graph.rows[e_mask]
            edges = edges[e_mask] 
            values = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.device)
            segment_add(e_mask.int(), graph.row_ptr, values)
            graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), values.cumsum(0, dtype=torch.int32)])
            in_windows_mask = in_windows_mask[e_mask]
            e_peeling_count = 0  
            edges_curr = torch.arange(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
            # buff_index = edges_curr[in_windows_mask]
            buff_index = torch.zeros(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
            buff = edges_curr[in_windows_mask]
            buf_size[0] = buff.shape[0]
            buff_index[:buf_size[0]] = buff
            n_mark = n_mark[e_mask]
            e_mask = torch.ones(graph.columns.shape[0], dtype=torch.bool, device=graph.device)
        e_curr = edges_curr[n_mark]
        if (e_peeling_count + e_curr.shape[0]) == graph.columns.shape[0]:  #如何正确跳出循环
            e_truss = torch.cat([e_truss, edges[e_curr]])
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
            break
        if e_curr.shape[0] == 0:
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.device)])
            l += 1
            if l == windows_high:
                windows_high = l+windows_skip
                in_windows_mask = (support >= l) & (support< windows_high) 
                # buff_index = edges_curr[in_windows_mask]
                buff = edges_curr[in_windows_mask]
                buf_size[0] = buff.shape[0]
                buff_index[:buf_size[0]] = buff
            # else:
            #     buff_index = edges_curr[in_windows_mask]   ###先试试这样吧，如果更慢了就想办法在cuda中智能添加buff_index
            # e_curr = torch.where(support == l)[0]  #也许这里
            e_curr = buff_index[:buf_size[0]][support[buff_index[:buf_size[0]]]==l]
            while e_curr.shape[0] == 0:
                ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
                l += 1
                if l == windows_high:
                    windows_high = l+windows_skip
                    in_windows_mask = (support >= l) & (support< windows_high) 
                    # buff_index = edges_curr[in_windows_mask]
                    buff = edges_curr[in_windows_mask]
                    buf_size[0] = buff.shape[0]
                    buff_index[:buf_size[0]] = buff
                # e_curr = buff_index[support[buff_index]==l] 
                e_curr = buff_index[:buf_size[0]][support[buff_index[:buf_size[0]]]==l]
    torch.cuda.synchronize()
    t22 = time.time()
    print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    return l+2, t11, t22

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

def test(graph: CSRCOO, n_cut, num_v):  
    #e_mask标记剩余边，
    #1. 修改核函数来标记inNext边, 使用cur_edge
    #2.用inwindow标记在索引范围内的边
    #3.每层结束后再压缩图
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
    graph.columns = graph.columns[edges]
    graph.rows = graph.rows[edges]
    graph.row_ptr = update_row_ptr(e_mask, graph.row_ptr)
    edges_curr = torch.arange(graph.columns.shape[0], dtype= torch.int32, device=graph.device)
    for i in range(1):
        torch.cuda.synchronize()
        start = time.time()
        e_curr = torch.where(support==l)[0]
        # graph.rows = graph.rows[e_mask]
        torch.cuda.synchronize()
        end = time.time()
        print(f"torch.where(support==l)[0]时间: {end - start:.6f} 秒")
    for i in range(1):
        torch.cuda.synchronize()
        start = time.time()
        e_curr = edges_curr[support==l]
        # e_curr = torch.gather_()
        torch.cuda.synchronize()
        end = time.time()
        print(f"edges_curr[support==l]时间: {end - start:.6f} 秒")
    

def main_csrcgraph(args):
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
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
    truss, t11, t22 = k_truss(graph, n_cut, num_v)
    # test(graph, n_cut, num_v)
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args() 
    # read_prepro_save(args)
    for i in range(1):
        main_csrcgraph(args)

   