"""
Impletation of Triangle Counting Algorithm using TCRGraph. 
"""
import math
import os
import sys
import torch
import argparse
import time
import logging
sys.path.append('/home/zhangqi/workspace/TCRGraph')
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection
import numpy as np
from torch_scatter import segment_csr
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
# 在计算支持度的时候，考虑负载均衡，并且存储三角形 
def truss_deposs(graph):
    #获取图的csr数据
    row_ptr = graph.row_ptr
    columns_g = graph.columns
    #计算出行索引号，和获取需要处理的无重复边，起点号小于终点号
    row_indice_g = torch.repeat_interleave(torch.arange(row_ptr.shape[0]-1, device=row_ptr.device), row_ptr[1:] - row_ptr[:-1])
    mask = row_indice_g < columns_g
    #计算边映射序号
    edges_id = torch.arange(columns_g.shape[0]//2, device=row_ptr.device)
    edges_id_nbr= torch.arange(columns_g.shape[0], device=row_ptr.device)
    edges_id_nbr[mask] = edges_id
    # print("edges_id_nbr[~mask]", edges_id_nbr[~mask])
    _, sorted_indices = torch.sort(columns_g[~mask], stable=True)
    # print("sorted_indices", sorted_indices)
    temp = torch.zeros_like(edges_id)
    temp[sorted_indices] = edges_id
    edges_id_nbr[~mask] = temp
    # print("edges_id_nbr", edges_id_nbr)
    #计算支持值  
    support, left_e, right_e, e_t_ptr, edges_id_c = edges_support_undirected(columns_g, row_ptr, edges_id_nbr, row_indice_g)   
    print("support[:30]", support[:30])                               
    l = 1
    sizes = e_t_ptr[1:]-e_t_ptr[:-1]
    mask_id = edges_id_c[sizes == 1] 
    # count = 0
    while True:
        # count = count + 1
        while mask_id.shape[0]==0:
            l = l + 1
            mask_id = edges_id_c[sizes == l]
            support[mask_id] = l
        # print("l", l)
        # print("mask_id", mask_id)
        mask = torch.repeat_interleave(torch.isin(edges_id_c, mask_id, invert=True), e_t_ptr[1:] - e_t_ptr[:-1])
        mask =  mask & torch.isin(left_e, mask_id, invert=True) & torch.isin(right_e, mask_id, invert=True)
        left_e = left_e[mask]
        right_e = right_e[mask]
        # print("left_e", left_e)
        mask1 = sizes > l
        sizes = segment_csr((mask).int(), e_t_ptr, reduce="sum")
        mask2 = sizes <= l
        mask_id = edges_id_c[mask1 & mask2]
        support[mask_id] = l     #这个是不是存在漏网之鱼？？？
        #其实已经没有三角形的边不用进入下一步处理
        mask = sizes > 0
        mask_id = edges_id_c[mask1 & mask2 & mask]
        #整理size和edges_id_c和e_t_ptr   ###以后要不要修改
        sizes = sizes[mask]
        edges_id_c = edges_id_c[mask]
        if edges_id_c.shape[0] == 0:
            break
        # print("edges_id_c", edges_id_c)
        e_t_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), sizes.cumsum(0)])
    return support+2

def edges_support_undirected(columns_g, row_ptr, edges_id_nbr, row_indice): 
    """
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
    # print("columns", columns)
    # print("row_indice", row_indice)
    # print("edges_id:", edges_id)
    left_e = torch.tensor([], dtype=torch.int64, device=row_ptr.device)
    right_e = torch.tensor([], dtype=torch.int64, device=row_ptr.device)
    #分批次计算
    max_values, _= torch.max(torch.stack((row_ptr[row_indice+1]-row_ptr[row_indice], row_ptr[columns+1]-row_ptr[columns]), dim=0), dim=0)
    max_values = max_values.to(torch.int64).cumsum(0)  #求和的值太大了，要检查是否溢出了
    batch = 1000000
    group = torch.searchsorted(max_values, torch.arange(0, max_values[-1]+batch, step=batch, dtype=torch.int64, device=row_ptr.device), side = 'right')
    group[0] = 0 
    off = 10000000
    del max_values
    for start, end in zip(group[0:-1], group[1:]):
        if start == end:
            break    
        s_c, s_ptr= batched_csr_selection(row_ptr[row_indice[start : end]], row_ptr[row_indice[start : end]+1])
        e_c, e_ptr= batched_csr_selection(row_ptr[columns[start : end]], row_ptr[columns[start : end]+1])
        #csr转为coo
        s_r = torch.repeat_interleave(torch.arange(s_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64)/off, s_ptr[1:] - s_ptr[:-1])
        e_r = torch.repeat_interleave(torch.arange(e_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64)/off, e_ptr[1:] - e_ptr[:-1])
        mask = torch.isin(s_r+columns_g[s_c], e_r+columns_g[e_c], assume_unique=True)
        # print("mask", mask)
        e_support[edges_id[start : end]] = segment_csr(mask.int(), s_ptr, reduce='sum')   #赋值只能用一层索引？？？？
        #左边
        left_e = torch.cat([left_e, edges_id_nbr[s_c][mask]])
        #右边
        mask = torch.isin(e_r+columns_g[e_c], s_r+columns_g[s_c], assume_unique=True)
        right_e = torch.cat([right_e, edges_id_nbr[e_c][mask]])
    #边csr指针ptr 够不成三角形的边没有放进去
    mask = e_support[edges_id] != 0
    e_t_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), e_support[edges_id[mask]].cumsum(0)])
    return e_support, left_e, right_e, e_t_ptr, edges_id[mask]

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
    support = truss_deposs(graph)
    # support = support_nocut_save(graph)
    t2 = time.time()
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    print('truss:{}'.format(support))
    sum = torch.max(support)
    print("k_max", sum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    # read_prepro_save(args)
    main_csrcgraph(args)
    # os.system('nvidia-smi')