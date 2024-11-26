"""
Impletation of Triangle Counting Algorithm using TCRGraph. 
"""
import os
import sys
import torch
import argparse
import time
import logging
sys.path.append('/home/zhangqi/workspace/TCRGraph')
from src.type.Graph import Graph
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
import numpy as np
from torch_scatter import segment_csr, scatter
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:3950"
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

#写两个函数，使用无向图数据的支持度计算 和 使用有向图数据的支持度计算

def edges_support_directed_nosave(columns_g, row_ptr, edges_id_nbr, row_indice): 
    """
    有向图数据，支持度计算优化
    返回值: e_support, left_e, right_e, e_ptr, edges_id
    """
    # #获取图的csr数据
    # row_ptr = graph.row_ptr
    # columns_g = graph.columns
    #建立存储支持度的张量
    e_support = torch.zeros(columns_g.shape[0]//2, dtype=torch.int32, device=row_indice.device)
    # #计算出每个顶点的邻居数量
    sizes = (row_ptr[1:] - row_ptr[:-1]) 
    # # 预处理，删除部分1-core的边
    # t3 = time.time()
    #这一步剪枝操作，希望能将减去的边进行标记
    while torch.any(sizes==1).item():
        non_leaf = torch.where(sizes != 1)[0]
        mask = torch.isin(columns_g, non_leaf) & torch.isin(row_indice, non_leaf)
        columns_g = columns_g[mask]
        row_indice = row_indice[mask]
        edges_id_nbr = edges_id_nbr[mask]
        sizes = segment_csr(mask.int(), row_ptr, reduce='sum')
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), sizes.cumsum(0)])
    # del non_leaf
    # t4 = time.time()
    # print('删除列Completed! {}s time elapsed. Outputting results...'.format(t4 - t3))
    #减去哪些边？？？将减去的边的支持度设为0
    mask = row_indice < columns_g
    columns = columns_g[mask]
    row_indice = row_indice[mask]
    edges_id = edges_id_nbr[mask]
    # half_row_ptr = segment_csr(mask.int(), row_ptr, reduce='sum')
    # print("columns", columns)
    # print("row_indice", row_indice)
    #分批次计算
    max_values = row_ptr[columns+1]-row_ptr[columns]
    max_values = max_values.to(torch.int64).cumsum(0)  #求和的值太大了，要检查是否溢出了
    batch = 10000000
    group = torch.searchsorted(max_values, torch.arange(0, max_values[-1]+batch, step=batch, dtype=torch.int64, device=row_ptr.device), side = 'right')
    del max_values
    group[0] = 0
    group = torch.unique(group)
    # torch.bucketize(half_row_ptr, group)
    off = 100000000
    for start, end in zip(group[0:-1], group[1:]): 
        rows, count = torch.unique(row_indice[start : end], return_counts=True)
        rows_ptr_cur = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), count.cumsum(0)])
        s_c, s_ptr= batched_csr_selection(row_ptr[rows], row_ptr[rows+1])
        e_c, e_ptr= batched_csr_selection(row_ptr[columns[start : end]], row_ptr[columns[start : end]+1])
        rows_ptr_cur = e_ptr[rows_ptr_cur]
        #csr转为coo
        s_r = torch.repeat_interleave(torch.arange(rows.shape[0], device=row_indice.device, dtype=torch.float64), s_ptr[1:] - s_ptr[:-1])
        e_r = torch.repeat_interleave(torch.arange(rows.shape[0], device=row_indice.device, dtype=torch.float64), rows_ptr_cur[1:]- rows_ptr_cur[:-1])
        # print("e_r", e_r)
        # print("rows.shape[0]", rows.shape[0])
        # print("s_ptr", s_ptr)  
        s_r = s_r + columns_g[s_c].to(torch.float64)/off
        e_r = e_r + columns_g[e_c].to(torch.float64)/off
        mask = torch.isin(e_r, s_r)
        # print("e_r+columns_g[e_c]", e_r)
        # print("s_r+columns_g[s_c]", s_r)
        # print("mask:", mask)
        e_support[edges_id[start : end]] = segment_csr(mask.int(), e_ptr, reduce='sum')
    return e_support

def support_nocut(graph: CSRGraph):  #无向图分批计算边支持度
    #稀疏图进行删边可以加速，紧密图不用进行删除边
    t8 = time.time()
    sizes_r = (graph.row_ptr[1:] - graph.row_ptr[:-1]) 
    ###################################根据每个顶点的邻居数量
    values = segment_csr(sizes_r[graph.columns], graph.row_ptr, reduce='sum')
    values = values.cumsum(0)
    print("values", values)
    batch = 100000000
    group = torch.searchsorted(values, torch.arange(0, values[-1]+batch-1, step=batch, dtype=torch.int64, device=graph.row_ptr.device), side = 'right')
    group[0] = 0
    group = torch.unique(group)
    print("group shape", group.shape[0])
    del sizes_r, values
    t9 = time.time()
    print('Completed! {}s time elapsed. Outputting results...'.format(t9 - t8))
    # group= torch.tensor([0, graph.num_vertices], dtype=torch.int64, device=graph.device)
    # edges_id = torch.arange(graph.columns.shape[0], dtype=torch.int64, device=graph.device)
    support = torch.zeros(graph.columns.shape[0], dtype=torch.int64, device=graph.device)
    off = 10000000
    for start, end in zip(group[:-1], group[1:]):
        vers = torch.arange(start, end, dtype=torch.int64, device=graph.row_ptr.device)
        #批量获取u邻居的csr格式
        u_c, u_ptr= batched_csr_selection_opt(graph.row_ptr[vers], graph.row_ptr[vers+1])
        # if u_ptr[-1] == 0:
        #     continue
        #csr转为coo
        u_r = torch.repeat_interleave(vers.to(torch.float64), u_ptr[1:] - u_ptr[:-1])
        #批量获取v的邻居
        v_c, v_ptr= batched_csr_selection_opt(graph.row_ptr[graph.columns[u_c]], graph.row_ptr[graph.columns[u_c]+1])
        u_ptr = v_ptr[u_ptr]
        v_r = torch.repeat_interleave(vers.to(torch.float64), u_ptr[1:] - u_ptr[:-1])
        v_r += (graph.columns[v_c]).to(torch.float64) / off
        u_r += (graph.columns[u_c]).to(torch.float64) / off
        t8 = time.time()
        mask = torch.isin(v_r, u_r)
        t9 = time.time()
        print("isin time:", t9-t8)
        #e1
        support[u_c] +=segment_csr(mask.int(), v_ptr, reduce='sum')
        #e2
        unique_e, counts = torch.unique(v_c[mask], return_counts=True)
        support[unique_e] += counts
        #e3
        indice = torch.bucketize(v_r[mask], u_r) 
        unique_e, counts = torch.unique(indice, return_counts=True)
        support[unique_e+graph.row_ptr[vers[0]]] += counts
    return support

def support_nocut_save(graph: CSRGraph):  #无向图分批计算边支持度，并存三角形
    #稀疏图进行删边可以加速，紧密图不用进行删除边
    sizes_r = (graph.row_ptr[1:] - graph.row_ptr[:-1]) 
    ####################################根据每个顶点的邻居数量
    values = segment_csr(sizes_r[graph.columns], graph.row_ptr, reduce='sum')
    values = values.cumsum(0)
    # print("values", values)
    batch = 10000000
    group = torch.searchsorted(values, torch.arange(0, values[-1]+batch-1, step=batch, dtype=torch.int64, device=graph.row_ptr.device), side = 'right')
    group[0] = 0
    group = torch.unique(group)
    del sizes_r, values
    # print("group len :", group.shape[0])
    # edges_id = torch.arange(graph.columns.shape[0], dtype=torch.int64, device=graph.device)
    support = torch.zeros(graph.columns.shape[0], dtype=torch.int64, device=graph.device)
    left_e = torch.tensor([], dtype=torch.int64, device=graph.device)
    right_e = torch.tensor([], dtype=torch.int64, device=graph.device)
    off = 10000000
    for start, end in zip(group[:-1], group[1:]):
        vers = torch.arange(start, end, dtype=torch.int64, device=graph.row_ptr.device)
        #批量获取u邻居的csr格式
        u_c, u_ptr= batched_csr_selection_opt(graph.row_ptr[vers], graph.row_ptr[vers+1])
        # if u_ptr[-1] == 0:
        #     continue
        #csr转为coo
        u_r = torch.repeat_interleave(vers.to(torch.float64), u_ptr[1:] - u_ptr[:-1])
        #批量获取v的邻居
        v_c, v_ptr= batched_csr_selection_opt(graph.row_ptr[graph.columns[u_c]], graph.row_ptr[graph.columns[u_c]+1])
        u_ptr = v_ptr[u_ptr]
        v_r = torch.repeat_interleave(vers.to(torch.float64), u_ptr[1:] - u_ptr[:-1])
        v_r += (graph.columns[v_c]).to(torch.float64) / off
        u_r += (graph.columns[u_c]).to(torch.float64) / off
        mask = torch.isin(v_r, u_r)
        #e1：u_c 重复segment_csr(mask.int(), v_ptr, reduce='sum')次
        support[u_c] +=segment_csr(mask.int(), v_ptr, reduce='sum')
        #e2：v_c[mask]
        left_e = torch.cat([left_e, v_c[mask]])
        # unique_e, counts = torch.unique(v_c[mask], return_counts=True)
        # support[unique_e] += counts
        #e3：indice
        indice = torch.bucketize(v_r[mask], u_r) 
        right_e = torch.cat([right_e, indice])
        # unique_e, counts = torch.unique(indice, return_counts=True)
        # support[unique_e] += counts
    size = support
    unique_e, counts = torch.unique(left_e, return_counts=True)
    support[unique_e] += counts
    unique_e, counts = torch.unique(right_e, return_counts=True)
    support[unique_e] += counts
    return support, left_e, right_e, size

def cut_leaf(graph: Graph):
    """
    一个图结构应该维护一个顶点列表、行偏移量、列值和边号
    但是剪枝并不可能减去一半数据，额外维护一个顶点mask或者顶点列表，开销好像比较大
    还是要额外新建一个truss分解的项目比较好
    所以不存三角形，分解时计算当前层的边的支持度时，还是得用到整个图数据。只有计算整个图的支持度，可以用一半的数据
    """
    pass

def read_prepro_save(args):
    print('reading graph...', end=' ', flush=True) 
    # graph, _= CSRGraph.read_graph(args.graph, directed=True)
    graph, _= CSRGraph.read_graph(args.graph)
    torch.save(graph, args.output)
    print('Saving Done!')
    return None

def main_csrcgraph(args):
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
   

    if args.cuda:
        graph.to('cuda:7')
        print('use cuda')

    t1 = time.time()
    # support = truss_deposs(graph)
    support = support_nocut(graph)
    t2 = time.time()
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    print('truss:{}'.format(support))
    # sum = torch.sum(support == 5)
    # print("sum", sum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    # read_prepro_save(args)
    # torch.set_num_threads(512)
    main_csrcgraph(args)
    # os.system('nvidia-smi')
    