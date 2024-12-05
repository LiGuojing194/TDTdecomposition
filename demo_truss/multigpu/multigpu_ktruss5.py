import pycuda.autoinit
from pycuda.compiler import SourceModule
import os
import torch
import argparse
import torch.distributed as dist
from torch.multiprocessing import Process
import time
import sys
sys.path.append('/root/autodl-tmp/TCRTruss32')
# from src.type.CSRCGraph_Truss import CSRCGraph
from src.type.CSRCOO import CSRCOO
from src.framework.helper import batched_csr_selection_opt
import numpy as np
import logging 
from torch_scatter import segment_csr
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
import torch.multiprocessing as mp

import viztracer 
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import SpectralClustering
# import community as community_louvain
import community.community_louvain as community_louvain
from scipy.cluster.hierarchy import linkage, fcluster

import segment_isin_extension  # 确保这与您的模块名称匹配
import segment_add_extension  # 确保这与您的模块名称匹配
# partiton
def replicate_edges_for_gpus(graph, num_gpus):
    row_ptr = graph.row_ptr
    columns_g = graph.columns
    print(row_ptr)
    data = np.ones(len(columns_g))
    number_of_nodes = len(row_ptr) - 1
    csr_matrix = sp.csr_matrix((data, columns_g, row_ptr), shape=(number_of_nodes, number_of_nodes))
    G = nx.convert_matrix.from_scipy_sparse_array(csr_matrix)

# 社区检测
    print("start")
    partition = community_louvain.best_partition(G)
    print("end")
    num_communities = len(set(partition.values()))

    # 构建超级节点图
    super_node_graph = nx.Graph()
    for node, community in partition.items():
        if community not in super_node_graph:
            super_node_graph.add_node(community)
        for neighbor in G.neighbors(node):
            neighbor_community = partition[neighbor]
            if neighbor_community != community:
                if super_node_graph.has_edge(community, neighbor_community):
                    super_node_graph[community][neighbor_community]['weight'] += 1
                else:
                    super_node_graph.add_edge(community, neighbor_community, weight=1)

    # 构建超级节点特征矩阵
    super_node_matrix = np.zeros((num_communities, num_communities))
    for i, j, data in super_node_graph.edges(data=True):
        super_node_matrix[i, j] = data['weight']
        super_node_matrix[j, i] = data['weight']

    # 层次聚类
    print("start")
    Z = linkage(super_node_matrix, 'ward')
    print("end")
    k = num_gpus
    super_node_labels = fcluster(Z, k, criterion='maxclust')
    final_partition = {}
    for node, community in partition.items():
        final_partition[node] = super_node_labels[community - 1]

    # 用户指定的聚类数目
 


# 输出每个社区的聚类标签

# 输出每个社区的聚类标签
    communities = {i: [] for i in range(1, k+1)}
    for node, label in final_partition.items():
        communities[label].append(node)
    community_tensor = torch.zeros(number_of_nodes, dtype=torch.int32)
    for node, label in final_partition.items():
        community_tensor[node] = label


  
    # 利用row_ptr生成所有边的起点索引
# 通过row_ptr相邻元素的差值得到每个节点的邻接节点数量，然后重复每个节点索引对应的次数
    #clusters_tensor = torch.tensor(clusters, dtype=torch.long)

# 利用 row_ptr 生成所有边的起点索引
    edge_start = torch.repeat_interleave(torch.arange(len(row_ptr) - 1, dtype=torch.long), torch.tensor(row_ptr[1:], dtype=torch.long) - torch.tensor(row_ptr[:-1], dtype=torch.long))

    # 所有边的终点索引直接由 columns_g 给出
    edge_end = torch.tensor(columns_g, dtype=torch.long)

    # 获取边起点和终点的社区标签
    start_labels = community_tensor[edge_start]
    end_labels = community_tensor[edge_end]

    # 比较起点和终点的社区标签，找出社区间的边
    inter_community_mask = start_labels != end_labels

    # 输出社区间的边的索引
    inter_community_edges_indices = inter_community_mask.nonzero(as_tuple=False).squeeze() 
    
    # 获取社区间边的两端点的度数
   # start_degrees = nodes_degrees[edge_start[inter_community_edges_indices]]
   # end_degrees = nodes_degrees[edge_end[inter_community_edges_indices]]

# 根据度数将边分配给度数较大的端点所在的社区
    #assigned_communities = torch.where(start_degrees > end_degrees, start_labels[inter_community_edges_indices], end_labels[inter_community_edges_indices])
    
# 转换社区标签为PyTorch张量
    

    # 生成随机数决定每条社区间边分配给起点或终点的社区
    random_assignments = torch.rand(inter_community_edges_indices.size(0)) > 0.5

    # 根据随机数结果分配社区
    assigned_communities = torch.where(random_assignments, community_tensor[edge_start[inter_community_edges_indices]], community_tensor[edge_end[inter_community_edges_indices]])

    # 更新社区节点集合
    for idx, community in zip(inter_community_edges_indices, assigned_communities):
        start_node = edge_start[idx].item()
        end_node = edge_end[idx].item()
        communities[community.item()].append(start_node)  # 添加起点到分配的社区
        communities[community.item()].append(end_node)    # 添加终点到分配的社区

# 确保社区中的节点是唯一的
    for community, nodes in communities.items():
        communities[community] = list(set(nodes))
        
        
    # 为每个社区重建CSR格式的图
    gpu_graphs_rowptr = []
    gpu_graphs_indices = []
    # 遍历社区，输出节点和边
    for community, nodes in communities.items():
        subgraph_nodes = set(nodes)  # 初始化为当前社区的节点集合

        # 遍历社区中的每个节点，只添加属于当前社区的邻居节点
        for node in nodes:
            start, end = row_ptr[node].item(), row_ptr[node + 1].item()
            neighbors = columns_g[start:end]
            # 添加直接邻居节点，不需要检查这些邻居的邻居
            subgraph_nodes.update(neighbors.tolist())
            # 对子图中的节点进行重新编号
        sorted_nodes = sorted(list(subgraph_nodes))
        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}

        # 构建新的CSR表示
        local_indices = []
        local_indptr = [0]

        for node in sorted_nodes:
            start, end = row_ptr[node].item(), row_ptr[node + 1].item()
            for neighbor in columns_g[start:end]:
                if neighbor.item() in subgraph_nodes:
                    local_indices.append(node_mapping[neighbor.item()])
            local_indptr.append(len(local_indices))

        gpu_graphs_indices.append(torch.tensor(local_indices))
        gpu_graphs_rowptr.append(torch.tensor(local_indptr))
    print(gpu_graphs_rowptr)
    print(gpu_graphs_indices)
    return gpu_graphs_indices, gpu_graphs_rowptr


def intersection(values, boundaries): #value和mask都有序
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


#################################################################################################################################
def support_computing(row_ptr,columns):
    #对数据进行分块
    sizes_r = (row_ptr[1:] - row_ptr[:-1]) 
    print("row_ptr type", row_ptr.dtype)
    print("sizes_r type", sizes_r.dtype)
    ####################################根据每个顶点的邻居数量，进行分块处理如何快速分块，
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    
    
    torch.cuda.synchronize()
    segment_add_extension.segment_add(sizes_r[columns], row_ptr, values)  #32位求和会溢出？
    torch.cuda.synchronize()
    
    
    values = values.cumsum(0)
    batch = 200000000
    group = torch.searchsorted(values, torch.arange(0, values[-1]+batch-1, step=batch, dtype=torch.int64, device=row_ptr.device), side = 'right')
    group[0] = 0
    group[-1] = values.shape[0]
    group = torch.unique(group)
    print("group shape", group.shape[0])
    
    torch.cuda.empty_cache()
    ######################################
    torch.cuda.synchronize()
    t11 = time.time()
    support = torch.zeros(columns.shape[0], dtype=torch.int32, device=row_ptr.device)
    row_ptr = row_ptr.to(torch.int32)
    for start, end in zip(group[:-1], group[1:]):
        #批量获取u邻居v的csr格式
        u_cs = row_ptr[start]
        u_ce = row_ptr[end]
        u_ptr = row_ptr[start:end+1]-row_ptr[start]  #放弃使用新变量u_ptr可以加速嘛，，ncut=1,,不切分时就不用使用u_ptr,就可以加速一点点。
        #u开始标记
        u_r = torch.repeat_interleave(torch.arange(end-start, dtype=torch.int32, device=row_ptr.device), u_ptr[1:] - u_ptr[:-1])
        #批量获取v的邻居w
        v_c, v_ptr= batched_csr_selection_opt(row_ptr[columns[u_cs: u_ce]], row_ptr[columns[u_cs: u_ce]+1])
        output = torch.zeros(v_ptr[-1], dtype =torch.int32, device=row_ptr.device)
        torch.cuda.synchronize()
        segment_isin_extension.segment_isin(v_ptr, columns[v_c], u_r, u_ptr, columns[u_cs: u_ce], output)  
        # torch.cuda.synchronize()
        #给三角形的三条边加上三角形数量
        # e2,e3, 或者用scatter函数，可以测试哪个处理得更快
        mask = output.to(torch.bool)
        unique_e, counts = torch.unique(output[mask] + u_cs, return_counts=True)
        support[unique_e] += counts
        unique_e, counts = torch.unique(v_c[mask], return_counts=True)
        support[unique_e] += counts
        # e1
        values = torch.zeros(v_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
        segment_add_extension.segment_add(mask.int().to(torch.int32), v_ptr, values)
        support[u_cs: u_ce] += values
    return support, t11

def sub_support(sub_columns, sub_row_ptr, sub_edges):   
    support = torch.zeros(sub_columns.shape[0], dtype=torch.int32, device=sub_columns.device)
    edges = torch.tensor([], dtype=torch.int32, device=sub_columns.device)
    left_e = torch.tensor([], dtype=torch.int32, device=sub_columns.device)
    right_e = torch.tensor([], dtype=torch.int32, device=sub_columns.device)
    group = torch.tensor([0, sub_row_ptr.shape[0]-1], dtype=torch.int32, device=sub_columns.device)
    for start, end in zip(group[0:-1], group[1:]):
        #批量获取u邻居v的csr格式
        u_cs = sub_row_ptr[start]
        u_ce = sub_row_ptr[end]
        u_ptr = sub_row_ptr[start:end+1]-sub_row_ptr[start]
        #u开始标记
        u_r = torch.repeat_interleave(torch.arange(end-start, dtype=torch.int32, device=sub_columns.device), u_ptr[1:] - u_ptr[:-1])
        #批量获取v的邻居w
        v_c, v_ptr= batched_csr_selection_opt(sub_row_ptr[sub_columns[u_cs: u_ce]], sub_row_ptr[sub_columns[u_cs: u_ce]+1])
        mask = torch.zeros(v_ptr[-1], dtype =torch.int32, device=sub_columns.device)
        segment_isin_extension.segment_isin(v_ptr, sub_columns[v_c], u_r, u_ptr, sub_columns[u_cs: u_ce], mask)  
        #给三角形的三条边加上三角形数量
        # e3
        right_e = torch.cat([right_e, sub_edges[mask[mask.to(torch.bool)] + u_cs]])
        # e2
        mask = mask.to(torch.bool)
        left_e = torch.cat([left_e, sub_edges[v_c[mask]]])
        # e1
        values = torch.zeros(v_ptr.shape[0]-1, dtype=torch.int32, device=sub_columns.device)
        segment_add_extension.segment_add(mask.int().to(torch.int32), v_ptr, values)
        torch.cuda.synchronize()
        support[u_cs: u_ce] += values
    edges = torch.repeat_interleave(sub_edges, support)
    return edges, left_e, right_e

def find_e_affected(p_s, p_e, e_curr, row_ptr,columns):
    #提取出当前边会影响到的子图
    p = torch.unique(torch.cat([p_s, p_e]))
    mask_v = torch.zeros(row_ptr.shape[0], dtype =torch.bool, device=row_ptr.device)
    mask_v[p] = True
    mask = mask_v[columns]  #python里索引最后一个就是-1
    p_c, _ = batched_csr_selection_opt(row_ptr[p], row_ptr[p+1])
    mask[p_c] = columns[p_c] != -1
    # mask_indice = torch.nonzero(mask).squeeze(1)
    # mask[mask_indice] = graph.columns[mask_indice] != -1 
    sub_columns = columns[mask]   #这里可以试试，用torch.nonzero(mask).squeeze(1)还是不用快
    sub_row_ptr = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    segment_add_extension.segment_add(mask.int().to(torch.int32), row_ptr, sub_row_ptr)
    sub_row_ptr = torch.cat([torch.zeros(1, dtype = torch.int32, device=row_ptr.device), sub_row_ptr.cumsum(0).to(torch.int32)])
    sub_edges = torch.arange(columns.shape[0], dtype = torch.int32, device=row_ptr.device)[mask]
    # print("sub_columns", sub_columns)
    # print("sub_row_ptr", sub_row_ptr)
    # print("sub_edges", sub_edges)
    #接下来找到子图中所有的三角形
    e1, e2, e3 = sub_support(sub_columns, sub_row_ptr, sub_edges)
    #如何用e2>-1来筛除-1，从而避免torch.isin的大开销 
    mask = intersection(e1, e_curr)   #edges和e_curr是顺序的所以是可以被优化的  ###时间瓶颈，有待优化#torch.isin很占空间。
    mask =  mask | intersection_nosorted(e2, e_curr) | intersection_nosorted(e3, e_curr)
    e_affected, a_counts = torch.unique(torch.cat([e1[mask], e2[mask], e3[mask]]), return_counts=True)  #没有剔除e_curr
    mask = intersection_invert(e_affected, e_curr)#e_affected和e_curr是有序的，#后续要对此进行优化
    return e_affected[mask], a_counts[mask].to(torch.int32)

def truss_deposs(row_ptr,columns):
    print(1)
    #计算支持值
    support, t11 = support_computing(row_ptr,columns)  
    print(2)
    #计算边映射序号             
    l = 1
    edges = torch.arange(columns.shape[0], device=row_ptr.device)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support.bool()
    columns = columns[mask]
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
    segment_add_extension.segment_add(mask.int().to(torch.int32), row_ptr, values)
    # torch.cuda.synchronize()
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), values.cumsum(0).to(torch.int32)])
    edges = edges[mask]
    e_curr = torch.where(support[edges]==l)[0]
    while e_curr.shape[0] == 0:
        l += 1
        e_curr = torch.where(support[edges]==l)[0] 
    while True:
        p_s = torch.bucketize(e_curr, row_ptr, right=True)-1  #使用bukitize查找，使用right=true这行查找的都大于1了
        p_e = columns[e_curr]
        e_affected, a_counts = find_e_affected(p_s, p_e, e_curr, row_ptr,columns)  #得找到受影响的边（不包括e_curr）及其重复次数
        #需要edges，来将edges和support对应起来。
        if e_affected.shape[0] != 0:        #********************得抽空测试测试这样能不能加速
            support[edges[e_affected]] -= a_counts    #？？？？？
            mask = support[edges[e_affected]] <= l
            support[edges[e_affected[mask]]] = l
            columns[e_curr] = -1
            e_curr = e_affected[mask]    #e_curr使用在压缩后的columns中的编号，而不使用具体的编号，用一个s_curr来指向当前的support中的相关值
        else:   #标记删除e_curr
            columns[e_curr] = -1
            e_curr = e_affected  
        ########################################
        if e_curr.shape[0] == 0:
            #进行一次子图计算, 先更新columns
            mask = columns != -1
            columns = columns[mask]
            if columns.shape[0] == 0:
                break
            #更新graph.row_ptr
            values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=row_ptr.device)
            segment_add_extension.segment_add(mask.int().to(torch.int32), row_ptr, values)
            row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=row_ptr.device), values.cumsum(0).to(torch.int32)])
            edges = edges[mask] 
            l += 1
            e_curr = torch.where(support[edges] == l)[0]
            while e_curr.shape[0] == 0:
                l += 1
                e_curr = torch.where(support[edges] == l)[0] 
    return support+2, t11
    
class mutilGPU_ktruss(Process):
    def __init__(self, rank, size,row_ptr,columns,**kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.size = size
        # 为每个进程设置特定的CUDA设备
        self.device = torch.device(f'cuda:{self.rank}')
        self.row_ptr = row_ptr
       
        self.columns = columns

    def setup(self, backend='nccl'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65535'
        dist.init_process_group(backend, rank=self.rank, world_size=self.size)



    def run(self):
         # 确保每个进程初始化其分布式组和设备
        self.setup()

        # 设置当前设备
        torch.cuda.set_device(self.device)
        

        # 消除初始化环境的影响
        self.row_ptr = self.row_ptr.to(self.device)
       
        self.columns = self.columns.to(self.device)
        #self.check()
        
        print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
        logging.info('num edges:{}.'.format(self.columns.shape[0]))
        
        
        # tracer = viztracer.VizTracer()
        # tracer.start()
        #I columns  D vertex_degrees II row_ptr K vertex_data
        truss, t11 = truss_deposs(self.row_ptr,self.columns)
        #torch.cuda.synchronize()
        """for edge_idx, (truss_value, u, v) in edge_to_truss_and_vertices.items():
            print("truss_value",truss_value)
            print("u",u)
            print("v",v)"""
        t2 = time.time()
        logging.info('rank {} finish and time {}'.format(self.rank, t2 - t11))
        



        
def read_prepro_save(args):
    print('reading graph...', end=' ', flush=True) 
    graph,_= CSRCOO.read_graph(args.graph)
    # graph, _= CSRGraph.read_graph(args.graph)
    #验证图的预处理的正确性
    print("graph.row_ptr", graph.row_ptr, graph.columns.dtype)
    print("graph.columns", graph.columns, graph.columns.dtype)
    # print("graph.column_ptr", graph.column_ptr, graph.column_ptr.dtype)
    # print("graph.rows", graph.rows, graph.rows.dtype)
    # print("graph.shuffle_ptr", graph.shuffle_ptr, graph.shuffle_ptr.dtype)
    torch.save(graph, args.output)
    print('Saving Done!')
    return None  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    
    #read_prepro_save(args)
    # main_csrcgraph(args)
    # graph = CSRCGraph.read_csrc_graph_bin(path)
    print("------------------------------------------------")
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
    print("args",args)
    """if args.cuda:
        graph.to('cuda')
        print('use cuda')"""
    # graph.row_ptr = graph.row_ptr.to(torch.int32)
    # graph.column_ptr = graph.column_ptr.to(torch.int32)
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    
    #row_ptrs, columns, vertex_begin_idx = get_partition(graph, num)
    
    
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError("This script requires at least 2 GPUs")
    columns,row_ptrs,=replicate_edges_for_gpus(graph, num_gpus)
    mp.set_start_method('spawn', force=True)
    processes = []

    for rank in range(num_gpus):
        row_ptr = torch.tensor(row_ptrs[rank]).to(torch.int32)
        print(row_ptr.shape[0])
        column = torch.tensor(columns[rank]).to(torch.int32)
        p = mutilGPU_ktruss(rank,num_gpus,row_ptr,column)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    