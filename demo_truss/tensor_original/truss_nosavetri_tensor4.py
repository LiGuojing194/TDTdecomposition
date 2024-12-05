"""
Impletation of Triangle Counting Algorithm using TCRGraph. 
"""
import os
import sys
import torch
import argparse
import time
import logging
sys.path.append('/root/autodl-tmp/TCRTruss32')
from src.type.Graph import Graph
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt2
import numpy as np
# from torch_scatter import segment_csr
from viztracer import VizTracer
from mytensorf import segment_add, segment_isin2, segment_isinis_m, sub_suppport_affect
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

"""
尝试优化这个无向图的truss分解算法
soc-orkut在取邻居的时候会溢出
tensor2的优化版
"""
def truss_deposs(graph):
    sizes = (graph.row_ptr[1:] - graph.row_ptr[:-1]) 
    row_indice_g = torch.repeat_interleave(torch.arange(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.row_ptr.device), sizes)
    print("row_indice_g:", row_indice_g)
    #获取一份无向图数据
    mask = row_indice_g < graph.columns
    row_indice = row_indice_g[mask]
    columns = graph.columns[mask]
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=graph.row_ptr.device)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device =graph.row_ptr.device)
    #生成张量e_mask来标记删除边
    e_mask = torch.ones(columns.shape[0], dtype=torch.bool, device =graph.row_ptr.device)
    #给去除1-core的图中的边进行编号
    edges_id = torch.arange(columns.shape[0], dtype=torch.int32, device=graph.row_ptr.device)
    edges_id_nbr= torch.arange(graph.columns.shape[0], dtype=torch.int32, device=graph.row_ptr.device)
    edges_id_nbr[mask] = edges_id
    _, sorted_indices = torch.sort(graph.columns[~mask], stable=True)
    temp = torch.zeros_like(edges_id)
    # print("temp type:", temp)
    temp[sorted_indices] = edges_id
    edges_id_nbr[~mask] = temp
    # print("edges_id_nbr", edges_id_nbr)
    ######################################################################
    #计算支持值  
    direc_row_ptr = torch.zeros(graph.row_ptr.shape[0], dtype=torch.int32, device =graph.row_ptr.device)
    segment_add(mask.int(), graph.row_ptr, direc_row_ptr)
    direc_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device =graph.row_ptr.device), direc_row_ptr.cumsum(0).to(torch.int32)])
    support = torch.zeros(columns.shape[0], dtype=torch.int32, device =graph.row_ptr.device)
    torch.cuda.synchronize()
    t1 = time.time()
    segment_isin2(row_indice, columns, direc_row_ptr, support) 
    print("support:", support)
    del direc_row_ptr
    ####################################################################
    #从这里开始计算分解时间
    #假设计算出来的是u<v的边的支持度
    #分解之前将支持度为0的边删除
    e_curr = edges_id[support == 0]
    e_mask[e_curr] = False
    # mask = torch.isin(edges_id_nbr, e_curr, invert=True)
    if not torch.any(e_mask):  
        return e_truss, ptr_truss
    mask = e_mask[edges_id_nbr]
    edges_id_nbr = edges_id_nbr[mask]
    graph.columns = graph.columns[mask]
    ######*******************replace******************################
    sizes = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.row_ptr.device)
    segment_add(mask.int(), graph.row_ptr, sizes)############
    ##########***********************************######################
    graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.row_ptr.device), sizes.cumsum(0).to(torch.int32)])  #全用32位
    #正式开始计算
    l = 1
    e_curr = torch.where(support == l)[0].to(torch.int32)
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.row_ptr.device)])
        l += 1
        e_curr = edges_id[support == l]
    e_del_count = 0
    graph.columns = torch.cat([graph.columns, torch.tensor([-1], dtype=torch.int32, device=graph.row_ptr.device)])
    while True:
        # print("l", l)
        logging.info('l:{}.'.format(l))
        print("e_curr shape:", e_curr.shape[0])
        e_truss = torch.cat([e_truss, edges_id[e_curr]])
        left_nbr, s_ptr = batched_csr_selection_opt2(graph.row_ptr[row_indice[e_curr]], graph.row_ptr[row_indice[e_curr]+1])   #这就是顶点对应剪枝后的图的编号
        right_nbr, e_ptr = batched_csr_selection_opt2(graph.row_ptr[columns[e_curr]], graph.row_ptr[columns[e_curr]+1])    
        ###############replace obtain mask1 and mask2#########################
        #分段求不是-1的交集，并返回两边相交元素的mask， 要剔除删除的边该怎么做？
        # e_mask_extend = e_mask[edges_id_nbr]
        # mask1  e_mask_extend[left_nbr]
        mask1 = e_mask[edges_id_nbr[left_nbr]]
        mask2 = e_mask[edges_id_nbr[right_nbr]]
        left_nbr[~mask1] = -1  
        right_nbr[~mask2] = -1  #这样不好，应该后面直接在columns上标记-1
        # torch.cuda.synchronize()
        mask1 = torch.zeros(left_nbr.shape[0], dtype=torch.bool, device=graph.row_ptr.device)
        mask2 = torch.zeros(right_nbr.shape[0], dtype=torch.bool, device=graph.row_ptr.device)
        # torch.cuda.synchronize()
        segment_isinis_m(graph.columns[left_nbr], s_ptr, graph.columns[right_nbr], e_ptr, mask1, mask2)
        # torch.cuda.synchronize()
        # print("torch.sum(mask1):", torch.sum(mask1))
        # print("torch.sum(mask2):", torch.sum(mask2))
        #################################################################################
        #删除边的三角形三边的获取
        left_nbr = edges_id_nbr[left_nbr[mask1]]
        right_nbr = edges_id_nbr[right_nbr[mask2]]
        if left_nbr.shape[0]==0 or right_nbr.shape[0] == 0:
            e_mask[e_curr] = False
            e_curr = left_nbr
        else:
            values = torch.zeros(e_ptr.shape[0]-1, dtype=torch.int32, device=graph.row_ptr.device)
            segment_add(mask2.int(), e_ptr, values)
            e_curr_rep = torch.repeat_interleave(e_curr, values)
            #################################################################
            mask1 = support[left_nbr] > l
            mask2 = support[right_nbr] > l
            #用cuda代码替代上面代码，返回一个张量记录处理过的边e_affacted
            e_affect_mask = torch.zeros(e_mask.shape[0], dtype=torch.bool, device=graph.row_ptr.device)
            sub_suppport_affect(left_nbr, right_nbr, e_curr_rep, support, mask1, mask2, e_affect_mask, l)
            #################################################################
            # print("e_affect_mask:", e_affect_mask)
            e_del_count += e_curr.shape[0]
            #进行删边标记, columns标记为-1的代价有点大，要不就算了
            # print("e_curr:", e_curr)
            # print("e_mask:", e_mask)
            e_mask[e_curr] = False
            # graph.columns = e_mask[edges_id_nbr]
            #更新下一轮处理的边号
            e_affect = torch.where(e_affect_mask)[0]
            mask = support[e_affect]<=l
            e_curr = e_affect[mask].to(torch.int32)
        #跳出最后一轮循环
        if torch.sum(e_mask) == e_curr.shape[0]:
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.row_ptr.device)])
            torch.cuda.synchronize()
            t2 = time.time()
            print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
            break
        #################################################################################################################
        #进行图压缩, 压缩columns_g、row_ptr, 但没有压缩support和e_mask
        if e_del_count >= 100000:
            print("---------------compress--------------------------------------------------")
            # print("e_mask:", e_mask)
            mask = e_mask[edges_id_nbr]
            edges_id_nbr = edges_id_nbr[mask]
            graph.columns = graph.columns[: -1][mask]
            graph.columns = torch.cat([graph.columns, torch.tensor([-1], dtype=torch.int32, device=graph.row_ptr.device)])
            sizes = torch.zeros(graph.row_ptr.shape[0]-1, dtype=torch.int32, device=graph.row_ptr.device)
            segment_add(mask.int(), graph.row_ptr, sizes)
            graph.row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.row_ptr.device), sizes.cumsum(0).to(torch.int32)])
            e_del_count = 0
            #为了简化前面在支持度上进行减值操作，有两种办法：1.不压缩support和e_mask 2.压缩support、e_mask，并更新edges_id_nbr
            #现在计划用方法2
            indice = torch.where(e_mask)[0]
            edges_id = edges_id[indice]     #edges_id这个变量可以不用的*****************************
            support = support[indice]
            columns = columns[indice]
            row_indice = row_indice[indice]
            nbrs_remap = torch.zeros(e_mask.shape[0], dtype=torch.int32, device=graph.row_ptr.device)
            nbrs_remap[indice] = torch.arange(indice.shape[0], dtype=torch.int32, device=graph.row_ptr.device)
            # print("nbrs_remap:", nbrs_remap)
            # print("e_curr:", e_curr)
            e_curr = nbrs_remap[e_curr]
            edges_id_nbr = nbrs_remap[edges_id_nbr]
            # print("e_curr:", e_curr)
            # print("edges_id_nbr:", edges_id_nbr)
            e_mask = torch.ones(indice.shape[0], dtype=torch.bool, device=graph.row_ptr.device)
            del indice, nbrs_remap
            # break
        #################################################################################################################
        #每层结束后, 找到下一层e_curr
        if e_curr.shape[0] == 0:
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.row_ptr.device)])
            l += 1
            e_curr = torch.where(support == l)[0]
            count = 0
            index = None
            while e_curr.shape[0] == 0:
                ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=graph.row_ptr.device)])
                l += 1
                if count % 100 == 0:
                    index = torch.where(support<(l+100))[0]
                    subsupport = support[index]
                e_curr = index[subsupport == l]
                count += 1
            e_curr = e_curr.to(torch.int32)
            # del subsupport
    return e_truss, ptr_truss


def main_csrcgraph():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
   
    print('reading graph...', end=' ', flush=True) 
    # graph, _= CSRGraph.read_graph(args.graph, directed=True)
    # graph, _= CSRGraph.read_graph(args.graph)
    # torch.save(graph, args.output)
    graph = torch.load(args.output)
    graph.pin_memory()
    print('Done!')

    if args.cuda:
        graph.to('cuda:0')
        print('use cuda')

    torch.cuda.synchronize()
    # t1 = time.time()
    # tracer = VizTracer()
    # tracer.start()
    e_truss, ptr_truss = truss_deposs(graph)
    # tracer.stop()
    # tracer.save()
    torch.cuda.synchronize()
    # t2 = time.time()
    # print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # os.system('nvidia-smi')
    print('e_truss :{}'.format(e_truss))
    print("truss", ptr_truss.shape[0]+1)
    


if __name__ == '__main__':
    main_csrcgraph()