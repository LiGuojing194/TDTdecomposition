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
from src.framework.helper import batched_csr_selection
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
import numpy as np
from torch_scatter import segment_csr
from viztracer import VizTracer
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

#不存三角形的Ktruss代码整理, 不存三角形且分批truss分解见Ktruss_v9.py
#设置next数组来分批计算
def truss_deposs(graph):
    #获取图的csr数据, 复制了一份，并不改变原始图
    row_ptr = graph.row_ptr
    columns_g = graph.columns
    #计算出行索引号，和获取需要处理的无重复边，起点号小于终点号
    sizes = (row_ptr[1:] - row_ptr[:-1]) 
    row_indice_g = torch.repeat_interleave(torch.arange(row_ptr.shape[0]-1, device=row_ptr.device), sizes)
    #进行剪枝操作，删除1-core节点和边 #这里只去除了边，没搞顶点，若是删了顶点数据，顶点就要重命名？？？
    non_leaf = None
    while torch.any(sizes==1).item():
        non_leaf = torch.where(sizes != 1)[0]
        mask = torch.isin(columns_g, non_leaf) & torch.isin(row_indice_g, non_leaf)
        columns_g = columns_g[mask]
        row_indice_g = row_indice_g[mask]
        sizes = segment_csr(mask.int(), row_ptr, reduce='sum')
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), sizes.cumsum(0)])
    del non_leaf
    mask = row_indice_g < columns_g
    row_indice = row_indice_g[mask]
    columns = columns_g[mask]
    #给去除1-core的图中的边进行编号
    edges_id = torch.arange(columns.shape[0], device=row_ptr.device)
    edges_id_nbr= torch.arange(columns_g.shape[0], device=row_ptr.device)
    edges_id_nbr[mask] = edges_id
    _, sorted_indices = torch.sort(columns_g[~mask], stable=True)
    temp = torch.zeros_like(edges_id)
    temp[sorted_indices] = edges_id
    edges_id_nbr[~mask] = temp
    # print("edges_id_nbr", edges_id_nbr)
    #计算支持值  
    support = edges_support_undirected(columns_g, row_ptr, row_indice, columns)
    #分解之前将支持度为0的边删除
    mask_id = edges_id[support == 0]
    mask = torch.isin(edges_id_nbr, mask_id, invert=True)
    if torch.all(~mask):  
        return support+2
    edges_id_nbr = edges_id_nbr[mask]
    columns_g = columns_g[mask]
    sizes = segment_csr(mask.int(), row_ptr, reduce="sum")
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), sizes.cumsum(0)])
    #正式开始计算
    l = 1
    threshold = 5000
    off = 10000
    next = edges_id[support == l]  #mask_id可以用torch.where函数代替
    if next.shape[0]<= threshold:
        mask_id = next
        next = torch.tensor([], dtype=torch.int64, device=row_ptr.device)
    else:
        mask_id = next[:threshold]
        next = next[threshold:]
    while True:
        while mask_id.shape[0]==0:
            # print("l",l)
            l += 1
            next = torch.cat([next, edges_id[support == l]])
            if next.shape[0]<= threshold:
                mask_id = next
                next = torch.tensor([], dtype=torch.int64, device=row_ptr.device)
            else:
                mask_id = next[:threshold]
                next = next[threshold:]
        print("l", l)
        #获取左右邻居在csr中的索引
        left_nbr, s_ptr = batched_csr_selection(row_ptr[row_indice[mask_id]], row_ptr[row_indice[mask_id]+1])   #这就是顶点对应剪枝后的图的编号
        right_nbr, e_ptr = batched_csr_selection(row_ptr[columns[mask_id]], row_ptr[columns[mask_id]+1])    
        s_r = torch.repeat_interleave(torch.arange(s_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64), s_ptr[1:] - s_ptr[:-1])
        e_r = torch.repeat_interleave(torch.arange(e_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64), e_ptr[1:] - e_ptr[:-1])
        columns_g = columns_g.to(torch.float64)
        s_r += columns_g[left_nbr] / off
        e_r += columns_g[right_nbr]/ off
        columns_g = columns_g.to(torch.int64)
        #左右两边邻居索引构成三角形的mask
        mask1 = torch.isin(s_r, e_r, assume_unique=True)
        mask2 = torch.isin(e_r, s_r[mask1], assume_unique=True)
        #  下面两行语句 6ms
        #找到三角形的左边边和右边边
        left_nbr = edges_id_nbr[left_nbr][mask1]
        right_nbr = edges_id_nbr[right_nbr][mask2]
        # mask_id_rep = torch.repeat_interleave(mask_id, e_ptr[1:] - e_ptr[:-1])[mask2]
        #peeling边对应的重复值
        mask_id_rep = mask_id[(e_r[mask2]).int()]
        # print("mask_id_rep", mask_id_rep)
        #这里应该可以用一个删除边张量做个标记
        mask1 = support[left_nbr] > l
        mask2 = support[right_nbr] > l
        sub_count = torch.zeros(support.shape[0], dtype=torch.int64, device=support.device) #是否必要？
        #mask1 & mask2 左右两条边都计数减一 
        unique_e, counts = torch.unique(left_nbr[mask1 & mask2], return_counts=True)
        sub_count[unique_e] = sub_count[unique_e] + counts
        unique_e, counts = torch.unique(right_nbr[mask1 & mask2], return_counts=True)
        sub_count[unique_e] = sub_count[unique_e] + counts
        #处理mask1&~mask2   8ms（建立索引更新索引较慢，要扫描所有元素）
        mask3 = mask1&(~mask2)
        mask4 = mask_id_rep[mask3]<right_nbr[mask3] 
        unique_e, counts = torch.unique(left_nbr[mask3][mask4], return_counts=True)
        sub_count[unique_e] = sub_count[unique_e] + counts
        #处理mask2&~mask1
        mask3 = (~mask1)&mask2
        mask4 = mask_id_rep[mask3]<left_nbr[mask3] #当前边号小，需要减一  
        unique_e, counts = torch.unique(right_nbr[mask3][mask4], return_counts=True)
        sub_count[unique_e] = sub_count[unique_e] + counts
        mask = (support >l)
        support = support - sub_count
        # print(support.dtype)
        # 数据矫正
        mask1 = mask & (support<=l)
        support[mask1] = l
        #删除边处理  #检查  #应该是将当前层的所有点删掉，而不是删除初始的mask_id
        mask = torch.isin(edges_id_nbr, mask_id, invert=True)
        if torch.all(~mask):  #
            break
        edges_id_nbr = edges_id_nbr[mask]
        columns_g = columns_g[mask]
        sizes = segment_csr(mask.int(), row_ptr, reduce="sum")
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=row_ptr.device), sizes.cumsum(0)])
        #下一轮处理的边号
        next = torch.cat([next, edges_id[mask1]])
        # print("next", next)
        if next.shape[0]<= threshold:
            mask_id = next
            next = torch.tensor([], dtype=torch.int64, device=row_ptr.device)
        else:
            mask_id = next[:threshold]
            next = next[threshold:]
    return support+2

def edges_support_undirected(columns_g, row_ptr, row_indice, columns):
    #分批次计算
    max_values, _= torch.max(torch.stack((row_ptr[row_indice+1]-row_ptr[row_indice], row_ptr[columns+1]-row_ptr[columns]), dim=0), dim=0)
    max_values = max_values.to(torch.int64).cumsum(0)  #求和的值太大了，要检查是否溢出了
    batch = 20000000
    group = torch.searchsorted(max_values, torch.arange(0, max_values[-1]+batch, step=batch, dtype=torch.int64, device=row_ptr.device), side = 'right')
    del max_values
    # print(len(group))
    # print("group[1:]-group[0:-1]", torch.max(group[1:]-group[0:-1]))
    off = 1000000
    #建立存储支持度的张量
    e_support = torch.zeros(columns.shape[0], dtype=torch.int64, device=row_indice.device)
    for start, end in zip(group[0:-1], group[1:]):
        if start == end:
            break
        s_c, s_ptr= batched_csr_selection(row_ptr[row_indice[start : end]], row_ptr[row_indice[start : end]+1])
        e_c, e_ptr= batched_csr_selection(row_ptr[columns[start : end]], row_ptr[columns[start : end]+1])
        s_c = columns_g[s_c]
        e_c = columns_g[e_c]
        #csr转为coo
        s_c = s_c + torch.repeat_interleave(torch.arange(s_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64)/off, s_ptr[1:] - s_ptr[:-1])
        e_c = e_c + torch.repeat_interleave(torch.arange(e_ptr.shape[0]-1, device=row_indice.device, dtype=torch.float64)/off, e_ptr[1:] - e_ptr[:-1])
        mask = torch.isin(s_c, e_c, assume_unique=True)
        e_support[start : end] = segment_csr(mask.int(), s_ptr, reduce='sum')
    return e_support




def main_csrcgraph():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
   
    print('reading graph...', end=' ', flush=True) 
    # graph, _= CSRGraph.read_graph(args.graph, directed=True)
    graph, _= CSRGraph.read_graph(args.graph)
    graph.pin_memory()
    print('Done!')

    if args.cuda:
        graph.to('cuda:0')
        print('use cuda')

    torch.cuda.synchronize()
    t1 = time.time()
    # tracer = VizTracer()
    # tracer.start()
    support = truss_deposs(graph)
    # tracer.stop()
    # tracer.save()
    torch.cuda.synchronize()
    t2 = time.time()
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # os.system('nvidia-smi')
    print('truss:{}'.format(support))
    sum = torch.sum(support == 5)
    print("sum", sum)
    


if __name__ == '__main__':
    main_csrcgraph()