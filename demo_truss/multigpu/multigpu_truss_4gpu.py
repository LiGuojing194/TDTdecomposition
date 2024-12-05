import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import sys
import torch
import argparse
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import psutil
sys.path.append('/root/autodl-tmp/TCRTruss32')
from src.type.Graph import Graph
from src.type.CSRCOO import CSRCOO
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection, batched_csr_selection_opt2
from mytensorf import segment_add, segment_isin2, segment_isin2tile, sub_AllAffectedSupport, sub_AllAffectedSupport_tile
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
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
def all_affect_support(e_affect, rows, columns, row_ptr, n_cut, mask,  l, n_mark, support):
    if n_cut>1:
        sub_AllAffectedSupport_tile(e_affect, rows, columns, row_ptr, n_cut, mask,  l, n_mark, support)
    else:
        sub_AllAffectedSupport(e_affect, rows, columns, row_ptr, mask,  l, n_mark, support)
###########################################################################
def k_truss_to_l(rows, columns, row_ptr, n_cut, num_v, l_max, support, sdevice):  #从l往上分解的算法设计
    # print("l_max:", l_max)
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    # support = support_computing(rows, columns, row_ptr, n_cut)
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=sdevice)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device = sdevice)
    #计算边映射序号             
    l = 1
    edges = torch.arange(columns.shape[0], device=sdevice)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support.bool()
    support = support[mask]
    columns = columns[mask]
    rows = rows[mask]
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
    segment_add(mask.int().to(torch.int32), row_ptr, values)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
    edges = edges[mask]
    e_curr = torch.where(support==l)[0]
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        if l == l_max:
            torch.cuda.synchronize()
            t22 = time.time()
            print("ptr_truss:", ptr_truss)
            print('First l truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
            return e_truss, ptr_truss, t11, t22
        e_curr = torch.where(support==l)[0]
    e_peeling_count = 0
    e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
    while True:
        # print("l", l)
        e_truss = torch.cat([e_truss, edges[e_curr]])
        p = torch.unique(rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v, dtype =torch.bool, device=sdevice)
        mask_v[p] = True 
        mask = mask_v[columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(row_ptr[p*n_cut],row_ptr[p*n_cut+n_cut])
        mask[p_c] = ~e_mask[p_c]
        #mask标记了需要查找三角形的边 从这里往下修改
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
        n_mark = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        #必须传递一个标记删除边
        all_affect_support(e_affect, rows, columns, row_ptr, n_cut, e_mask, l, n_mark, support)
        columns[e_curr] = -1   #看看能不能把这行去掉
        # support[e_curr] = l #增加一行矫正
        e_peeling_count += e_curr.shape[0]
        if e_peeling_count > 1000000:
            e_mask = ~e_mask
            support = support[e_mask]
            columns = columns[e_mask]
            rows = rows[e_mask]
            values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
            segment_add(e_mask.int().to(torch.int32), row_ptr, values)
            row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
            edges = edges[e_mask] 
            e_peeling_count = 0  
            e_curr = torch.where(support <= l)[0]
            e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        else:
            e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
            n_mark = support[e_curr]<=l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
            e_curr = e_curr[n_mark]
        while e_curr.shape[0] == 0:
            # print("First l truss decomposition ptr_truss:, l_max:", l, ptr_truss, l_max)
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=sdevice)])
            l += 1
            if l == l_max:
                torch.cuda.synchronize()
                t22 = time.time()
                # print("First l truss decomposition ptr_truss:", ptr_truss)
                # print('First l truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
                # logging.info('First l truss decomposition ptr_truss: {} '.format(ptr_truss))
                logging.info('First l truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
                return e_truss, ptr_truss, t11, t22
            e_curr = torch.where(support == l)[0] 
    torch.cuda.synchronize()
    t22 = time.time()
    return e_truss, ptr_truss, t11, t22
#############################################################################
def k_truss_from_l(rows, columns, row_ptr, n_cut, l_max, support, sdevice):  
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    # support = support_computing(rows, columns, row_ptr, n_cut)
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device = sdevice)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device = sdevice)
    #计算边映射序号             
    l = l_max
    edges = torch.arange(columns.shape[0], device=sdevice)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support >= l
    # support = support[mask]
    columns = columns[mask]
    rows = rows[mask]
    edges = edges[mask]
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
    segment_add(mask.int().to(torch.int32), row_ptr, values)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
    #顶点重编号 #如果删除的边数量大于100万再进行顶点重编号
    if (mask.shape[0] - columns.shape[0]) > 1000000:
        values = values.view(-1, n_cut)
        e_mask = torch.sum(values.bool(), dim=1)
        e_mask[torch.unique(columns)] = True
        s_index = torch.nonzero(e_mask).squeeze(1)  #这下面的顶点重新编号，还涉及分块要考虑  #为什么分块数为1时，计算结果还是错的呢？忽略了没有更大邻居的columns
        num_v = s_index.shape[0]
        row_ptr = torch.flatten(values[s_index])
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), row_ptr.cumsum(0).to(torch.int32)]) #所以row_ptr的开头是[0, 0]
        values = values[:, 0] 
        values[s_index] = torch.arange(0, s_index.shape[0], dtype=torch.int32, device=sdevice) #从1开始编号的
        rows = values[rows]
        columns = values[columns]
    else:
        num_v = row_ptr.shape[0]
    #重新计算支持度
    support = support_computing(rows, columns, row_ptr, n_cut)
    #找到要删除的边
    e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
    e_curr = torch.where(support<l)[0]
    #将不属于 l truss 的边都标记并删除
    while e_curr.shape[0] != 0:
        p = torch.unique(rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v+1, dtype =torch.bool, device=sdevice)
        mask_v[p] = True 
        mask = mask_v[columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(row_ptr[p*n_cut], row_ptr[p*n_cut+n_cut])
        mask[p_c] = ~e_mask[p_c]
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #看看能不能改代码，将e_mask标记成剩余边
        n_mark = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        all_affect_support(e_affect, rows, columns, row_ptr, n_cut, e_mask, l, n_mark, support)
        columns[e_curr] = -1 
        e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
        n_mark = support[e_curr]<l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
        e_curr = e_curr[n_mark]
    e_curr = torch.where(support==l)[0]
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        e_curr = torch.where(support==l)[0]
    e_peeling_count = torch.sum(e_mask)
    while True:
        e_truss = torch.cat([e_truss, edges[e_curr]])
        p = torch.unique(rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v, dtype =torch.bool, device=sdevice)
        mask_v[p] = True 
        mask = mask_v[columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(row_ptr[p*n_cut], row_ptr[p*n_cut+n_cut])
        mask[p_c] = ~e_mask[p_c]
        #mask标记了需要查找三角形的边 从这里往下修改
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
        n_mark = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        #必须传递一个标记删除边
        all_affect_support(e_affect, rows, columns, row_ptr, n_cut, e_mask, l, n_mark, support)
        columns[e_curr] = -1   #看看能不能把这行去掉
        # support[e_curr] = l #增加一行矫正
        e_peeling_count += e_curr.shape[0]
        if e_peeling_count > 1000000:
            e_mask = ~e_mask
            support = support[e_mask]
            columns = columns[e_mask]
            rows = rows[e_mask]
            values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
            segment_add(e_mask.int().to(torch.int32), row_ptr, values)
            row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
            edges = edges[e_mask] 
            e_peeling_count = 0  
            e_curr = torch.where(support <= l)[0]
            e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        else:
            e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
            n_mark = support[e_curr]<=l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
            e_curr = e_curr[n_mark]
        if (e_peeling_count + e_curr.shape[0]) == columns.shape[0]:  #如何正确跳出循环
            e_truss = torch.cat([e_truss, edges[e_curr]])
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=sdevice)])
            break
        if e_curr.shape[0] == 0:
            # print("before l:", l)
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=sdevice)])
            l += 1
            e_curr = torch.where(support == l)[0]  #也许这里
            while e_curr.shape[0] == 0:
                ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
                l += 1
                e_curr = torch.where(support == l)[0] 
    torch.cuda.synchronize()
    t22 = time.time()
    logging.info('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    # print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    return l+2, t11, t22

def k_truss_from_l_to_l(rows, columns, row_ptr, n_cut, l_min, l_max, support, sdevice):  
    # print("l_max:", l_max)
    torch.cuda.synchronize()
    t11 = time.time()
    #计算支持值
    # support = support_computing(rows, columns, row_ptr, n_cut)
    #生成空的e_truss来存放peeling了的边
    e_truss = torch.tensor([], dtype=torch.int32, device=sdevice)
    ptr_truss = torch.zeros(1, dtype=torch.int32, device=sdevice)
    #计算边映射序号             
    l = l_min
    edges = torch.arange(columns.shape[0], device=sdevice)
    #第一步，整理整个图，支持度为零的数据清除
    mask = support >= l
    # support = support[mask]
    columns = columns[mask]
    rows = rows[mask]
    edges = edges[mask]
    values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
    segment_add(mask.int().to(torch.int32), row_ptr, values)
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
    #顶点重编号
    values = values.view(-1, n_cut)
    e_mask = torch.sum(values.bool(), dim=1)
    e_mask[torch.unique(columns)] = True
    s_index = torch.nonzero(e_mask).squeeze(1)  #这下面的顶点重新编号，还涉及分块要考虑  #为什么分块数为1时，计算结果还是错的呢？忽略了没有更大邻居的columns
    num_v = s_index.shape[0]
    row_ptr = torch.flatten(values[s_index])
    row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), row_ptr.cumsum(0).to(torch.int32)]) #所以row_ptr的开头是[0, 0]
    values = values[:, 0] 
    values[s_index] = torch.arange(0, s_index.shape[0], dtype=torch.int32, device=sdevice) #从1开始编号的
    rows = values[rows]
    columns = values[columns]
    #重新计算支持度
    support = support_computing(rows, columns, row_ptr, n_cut)
    #找到要删除的边
    e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
    e_curr = torch.where(support<l)[0]
    #将不属于 l truss 的边都标记并删除
    while e_curr.shape[0] != 0:
        p = torch.unique(rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v+1, dtype =torch.bool, device=sdevice)
        mask_v[p] = True 
        mask = mask_v[columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(row_ptr[p*n_cut], row_ptr[p*n_cut+n_cut])
        mask[p_c] = ~e_mask[p_c]
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #看看能不能改代码，将e_mask标记成剩余边
        n_mark = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        all_affect_support(e_affect, rows, columns, row_ptr, n_cut, e_mask, l, n_mark, support)
        columns[e_curr] = -1 
        # mask = (support<l) & (~e_mask)
        # e_curr = torch.where(mask)[0] 
        e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
        n_mark = support[e_curr]<l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
        e_curr = e_curr[n_mark]
        # print("e_curr", e_curr)
    e_curr = torch.where(support==l)[0]
    while e_curr.shape[0] == 0:
        ptr_truss = torch.cat([ptr_truss, ptr_truss[-1].unsqueeze(0)])
        l += 1
        if l == l_max:
            torch.cuda.synchronize()
            t22 = time.time()
            print("ptr_truss:", ptr_truss)
            print('First 2 truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
            return e_truss, ptr_truss, t11, t22
        e_curr = torch.where(support==l)[0]
    e_peeling_count = torch.sum(e_mask)
    # e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
    while True:
        e_truss = torch.cat([e_truss, edges[e_curr]])
        p = torch.unique(rows[e_curr]) #这里面就不该有-1
        mask_v = torch.zeros(num_v, dtype =torch.bool, device=sdevice)
        mask_v[p] = True 
        mask = mask_v[columns]  #python里索引最后一个就是-1
        p_c, _ = batched_csr_selection_opt2(row_ptr[p*n_cut], row_ptr[p*n_cut+n_cut])
        mask[p_c] = ~e_mask[p_c]
        #mask标记了需要查找三角形的边
        e_affect = torch.nonzero(mask).squeeze(1).to(torch.int32) 
        e_mask[e_curr] = True #标记了待删的e_curr, 包括当前这轮要删除的边
        n_mark = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        #必须传递一个标记删除边
        all_affect_support(e_affect, rows, columns, row_ptr, n_cut, e_mask, l, n_mark, support)
        columns[e_curr] = -1   #看看能不能把这行去掉
        # support[e_curr] = l #增加一行矫正
        e_peeling_count += e_curr.shape[0]
        if e_peeling_count > 1000000:
            e_mask = ~e_mask
            support = support[e_mask]
            columns = columns[e_mask]
            rows = rows[e_mask]
            values = torch.zeros(row_ptr.shape[0]-1, dtype=torch.int32, device=sdevice)
            segment_add(e_mask.int().to(torch.int32), row_ptr, values)
            row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=sdevice), values.cumsum(0).to(torch.int32)])
            edges = edges[e_mask] 
            e_peeling_count = 0  
            e_curr = torch.where(support <= l)[0]
            e_mask = torch.zeros(columns.shape[0], dtype=torch.bool, device=sdevice)
        else:
            e_curr = torch.nonzero(n_mark).squeeze(1)  #n_mark不会标记删除过的边，所以不用矫正
            n_mark = support[e_curr]<=l  #其实可以测试，是直接判断support[~e_mask]<=l快，还是利用n_mark标记快
            e_curr = e_curr[n_mark]
        while e_curr.shape[0] == 0:
            ptr_truss = torch.cat([ptr_truss, torch.tensor([e_truss.shape[0]], dtype=torch.int32, device=sdevice)])
            l += 1
            if l == l_max:
                torch.cuda.synchronize()
                t22 = time.time()
                # print("ptr_truss:", ptr_truss)
                # logging.info('ptr_truss: {} '.format(ptr_truss))
                logging.info('From l to l truss decomposition Completed! {}s time elapsed. Outputting results...'.format(t22 - t11))
                return e_truss, ptr_truss, t11, t22
            e_curr = torch.where(support == l)[0] 
    torch.cuda.synchronize()
    t22 = time.time()
    print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    return l+2, t11, t22

##############################################################################
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
        p_c, _ = batched_csr_selection_opt2(graph.row_ptr[p*n_cut], graph.row_ptr[p*n_cut+n_cut])
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
        all_affect_support(e_affect, graph.rows, graph.columns, graph.row_ptr, n_cut, e_mask, l, n_mark, support)
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
            # print("before l:", l)
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
    print('{} truss decomposition Completed! {}s time elapsed. Outputting results...'.format(l+2, t22 - t11))
    return l+2, t11, t22
#################################################################################
def find_split_l(graph, n_cut, num_hpus): #1. 指数探索+二分查找 找到l; 2. 要分割数据吗？  #或者通过k_core_max来分割
    """
    尽可能均分,先除去0这一层
    """
    l_split = torch.zeros(num_hpus-1, dtype=torch.int32, device=graph.device)
    l = 1
    half = graph.columns.shape[0]//3*2
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)
    temp_count = 0
    for i in range(num_hpus-1):
        while temp_count < half:
            l = l*2
            temp_count = torch.sum(support<l)
        left = l//2
        right = l
        while left < right:
            # print("l:", l)
            l = (left + right)//2
            temp_count = torch.sum(support<l)
            if temp_count < half:
                left = l+1
            else:
                right = l
        l = right
        if i == 0:
            l_split[i] = l if l > 2 else 2
        else:
            l_split[i] = l if l>l_split[i-1] else l_split[i-1]+2
        # support = support[support>=l]
        # temp_count = 0
        half += half//3
    print("The found l is :", l_split)
    return l_split, support   #support被变化了

def find_split_l_del0(graph, n_cut, num_hpus): #1. 指数探索+二分查找 找到l; 2. 要分割数据吗？  #或者通过k_core_max来分割
    """
    尽可能均分,先除去0这一层, 可能还得加一个k_core最大值的估计函数
    """
    l_split = torch.zeros(num_hpus-1, dtype=torch.int32, device=graph.device)
    l = 1
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)
    subsupport = support[support.bool()]
    half = subsupport.shape[0]//3*2
    temp_count = 0
    for i in range(num_hpus-1):
        while temp_count < half:
            l = l*2
            temp_count = torch.sum(subsupport<l)
        left = l//2
        right = l
        while left < right:
            # print("l:", l)
            l = (left + right)//2
            temp_count = torch.sum(subsupport<l)
            if temp_count < half:
                left = l+1
            else:
                right = l
        l = right
        if i == 0:
            l_split[i] = l if l > 2 else 2
        else:
            l_split[i] = l if l>l_split[i-1] else l_split[i-1]+2
        subsupport = subsupport[subsupport>=l]
        temp_count = 0
        half = half//3
    print("The found l is :", l_split)
    return l_split, support   #support被变化了

def find_split_l2(graph, n_cut, num_hpus): #1. 指数探索+二分查找 找到l; 2. 要分割数据吗？  #或者通过k_core_max来分割
    """
    将三角形数量尽可能均分, 这样分任务1会的工作会更多
    """
    l_split = torch.zeros(num_hpus-1, dtype=torch.int32, device=graph.device)
    l = 1
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)  #以gpu和cpu计算支持度的时间比来衡量计算能力
    half = torch.sum(support)//(num_hpus)
    temp_count = 0
    for i in range(num_hpus-1):
        while temp_count < half:
            l = l*2
            index = torch.where(support<l)[0]
            temp_count = torch.sum(support[index])
        left = l//2
        right = l
        while left < right:
            l = (left + right)//2
            print("l:", l)
            # temp_count = torch.sum(support<l)
            index = torch.where(support<l)[0]
            temp_count = torch.sum(support[index])
            if temp_count < half:
                left = l+1
            else:
                right = l
        l = right
        if i == 0:
            l_split[i] = l if l > 2 else 2
        else:
            l_split[i] = l if l>l_split[i-1] else l_split[i-1]+2
    #     l_split[i] = 8
    # l_split[0] = 4
    # l_split[1] = 8
    # l_split[2] = 16
    # support = support[support>=l]
    # temp_count = 0
        half += half
    print("The found l is :", l_split)
    return l_split, support   #support被变化了

def find_split_edgetri(graph, n_cut, num_hpus): #1. 指数探索+二分查找 找到l; 2. 要分割数据吗？  #或者通过k_core_max来分割
    """
   先试试分成2个GPU的函数管不管用
    """
    l_split = torch.zeros(num_hpus-1, dtype=torch.int32, device=graph.device)
    l = 1
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)  #以gpu和cpu计算支持度的时间比来衡量计算能力
    tri_sum = torch.sum(support)
    total_enum = support.shape[0]
    index = torch.where(support<l)[0]
    gpu1_enum = index.shape[0]
    rest_enum = total_enum - gpu1_enum
    temp_count = torch.sum(support[index])
    curr_value = (1+0.3*total_enum/rest_enum)*temp_count
    while curr_value < tri_sum:
        l = l*2
        index = torch.where(support<l)[0]
        gpu1_enum = index.shape[0]
        rest_enum = total_enum - gpu1_enum
        temp_count = torch.sum(support[index])
        curr_value = (1+0.3*total_enum/rest_enum)*temp_count
        left = l//2
        right = l
        while left < right:
            l = (left + right)//2
            print("l:", l)
            index = torch.where(support<l)[0]
            gpu1_enum = index.shape[0]
            rest_enum = total_enum - gpu1_enum
            temp_count = torch.sum(support[index])
            curr_value = (1+0.3*total_enum/rest_enum)*temp_count
            if curr_value < tri_sum:
                left = l+1
            else:
                right = l
        l = right
        # if i == 0:
        #     l_split[i] = l if l > 2 else 2
        # else:
        #     l_split[i] = l if l>l_split[i-1] else l_split[i-1]+2
    l_split[0] = l
    print("The found l is :", l_split)
    return l_split, support  

def find_split_edgetri2(graph, n_cut, num_hpus): #1. 指数探索+二分查找 找到l; 2. 要分割数据吗？  #或者通过k_core_max来分割
    """
   先试试分成2个GPU的函数管不管用
    """
    l_split = torch.zeros(num_hpus-1, dtype=torch.int32, device=graph.device)
    l = 1
    support = support_computing(graph.rows, graph.columns, graph.row_ptr, n_cut)  #以gpu和cpu计算支持度的时间比来衡量计算能力
    tri_sum = torch.sum(support)
    total_enum = support.shape[0]
    index = torch.where(support<l)[0]
    gpu1_enum = index.shape[0]
    rest_enum = total_enum - gpu1_enum
    temp_count = torch.sum(support[index])
    curr_value = (1 + gpu1_enum/rest_enum)*temp_count
    while curr_value < tri_sum:
        l = l*2
        index = torch.where(support<l)[0]
        gpu1_enum = index.shape[0]
        rest_enum = total_enum - gpu1_enum
        temp_count = torch.sum(support[index])
        curr_value = (1+gpu1_enum/rest_enum)*temp_count
        left = l//2
        right = l
        while left < right:
            l = (left + right)//2
            print("l:", l)
            index = torch.where(support<l)[0]
            gpu1_enum = index.shape[0]
            rest_enum = total_enum - gpu1_enum
            temp_count = torch.sum(support[index])
            curr_value = (1+gpu1_enum/rest_enum)*temp_count
            if curr_value < tri_sum:
                left = l+1
            else:
                right = l
        l = right
        # if i == 0:
        #     l_split[i] = l if l > 2 else 2
        # else:
        #     l_split[i] = l if l>l_split[i-1] else l_split[i-1]+2
    l_split[0] = l
    print("The found l is :", l_split)
    return l_split, support   #support被变化了

def csr_to_tilingcsr(graph: CSRCOO, tiling, n_cut):
    tiling_row_ptr = torch.zeros(graph.num_vertices*n_cut, dtype=torch.int32, device=graph.device)
    tiling_block = graph.columns//(tiling+1) + graph.rows*n_cut
    print("tiling_block", tiling_block)
    e_u, e_counts = torch.unique_consecutive(tiling_block, return_counts = True)
    tiling_row_ptr[e_u] = e_counts.to(torch.int32)
    tiling_row_ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=graph.device), tiling_row_ptr.cumsum(0).to(torch.int32)])
    return tiling_row_ptr

class mutilGPU_ktruss(Process):
    def __init__(self, rank, size, rows, columns, row_ptr, n_cut, num_v, l_split, support, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.size = size
        # 为每个进程设置特定的CUDA设备
        self.device = torch.device(f'cuda:{self.rank}')
        # self.graph = graph
        self.rows = rows
        self.columns = columns
        self.row_ptr = row_ptr
        self.n_cut = n_cut
        self.num_v = num_v
        self.l_split = l_split
        self.support = support

    def setup(self, backend='nccl'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '65535'
        dist.init_process_group(backend, rank=self.rank, world_size=self.size)

    def run(self):
         # 确保每个进程初始化其分布式组和设备
        self.setup()
        # 设置当前设备
        gpu_id = dist.get_rank()
        if self.rank == 0:
            print(f"Rank: {self.rank}, World Size: {dist.get_world_size()}")
            torch.cuda.set_device(self.device)
            # 消除初始化环境的影响
            self.row_ptr = self.row_ptr.to(self.device)
            self.columns = self.columns.to(self.device)
            self.rows =  self.rows.to(self.device)
            self.support = self.support.to(self.device)
            # self.l_split = self.l_split.to(self.device)
            print("suppport device:", self.support.device)
            k_truss_to_l(self.rows, self.columns, self.row_ptr,  self.n_cut, self.num_v, self.l_split[0].item(), self.support, self.device)
        elif self.rank == (self.size-1):
            print(f"Rank: {self.rank}, World Size: {dist.get_world_size()}")
            torch.cuda.set_device(self.device)
            # 消除初始化环境的影响
            self.row_ptr = self.row_ptr.to(self.device)
            self.columns = self.columns.to(self.device)
            self.rows =  self.rows.to(self.device)
            self.support = self.support.pin_memory()
            self.support = self.support.to(self.device)
            # self.l_split = self.l_split.to(self.device)
            print("suppport device:", self.support.device)
            k_truss_from_l(self.rows, self.columns, self.row_ptr, self.n_cut, self.l_split[-1].item(), self.support, self.device)
        else:
            print(f"Rank: {self.rank}, World Size: {dist.get_world_size()}")
            torch.cuda.set_device(self.device)
            # 消除初始化环境的影响
            self.row_ptr = self.row_ptr.to(self.device)
            self.columns = self.columns.to(self.device)
            self.rows =  self.rows.to(self.device)
            self.support = self.support.to(self.device)
            # self.l_split = self.l_split.to(self.device)
            k_truss_from_l_to_l(self.rows, self.columns, self.row_ptr, self.n_cut, self.l_split[gpu_id-1].item(), self.l_split[gpu_id].item(), self.support, self.device)

        
        # logging.info('num edges:{}.'.format(self.columns.shape[0]))


def main_csrcgraph(args):
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
    
    if args.cuda:
        graph.to('cuda')
        print('use cuda')
    # print("graph.rows", graph.rows)
    # print("graph.columns", graph.columns)
    # print("graph.ptr", graph.row_ptr)
    n_cut = 2
    if n_cut > 1:
        tiling = graph.num_vertices // n_cut
        graph.row_ptr = csr_to_tilingcsr(graph, tiling, n_cut)
    
    num_hpus = torch.cuda.device_count()+1
    # l_split, support = find_split_edgetri2(graph, n_cut, 2)
    # find_split_l_del0(graph, n_cut, 2)
    # find_split_l(graph, n_cut, 2)
    l_split, support = find_split_l2(graph, n_cut, num_hpus)
    # support = support.pin_memory()
    
    num_v = graph.num_vertices
    # # print("graph.device", graph.device)
    # # print("support", support)
    # # k_truss_to_l(graph, n_cut, num_v, l_split[0], support)
    # # k_truss_from_l(graph, n_cut, 4, support)
    # # truss, t11, t22 = k_truss(graph, n_cut, num_v)
    # l_split[0] =l_split[0] // 2
    # l_split[0] = 1
    k_truss_to_l(graph.rows, graph.columns, graph.row_ptr,  n_cut, num_v, l_split[0].item(), support, graph.device)
    k_truss_from_l(graph.rows, graph.columns, graph.row_ptr,  n_cut, l_split[0].item(), support, graph.device)

    # mp.set_start_method('spawn', force=True)
    # processes = []
    # # l_split[0] = 3
    # for rank in range(num_hpus):
    #     p = mutilGPU_ktruss(rank, num_hpus, graph.rows, graph.columns, graph.row_ptr, n_cut, num_v, l_split, support)  #rank, size, graph,
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

def read_prepro_save(args):
    print('reading graph...', end=' ', flush=True) 
    graph, _= CSRCOO.read_graph(args.graph, directed=True)
    print(graph.row_ptr.dtype)
    torch.save(graph, args.output)
    print('Saving Done!')
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args() 
    # read_prepro_save(args)
    for i in range(1):
        main_csrcgraph(args)