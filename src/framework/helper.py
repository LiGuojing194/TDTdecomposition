"""
Helper functions used in the TCRGraph framework.
"""
from typing import Optional, Tuple
import torch
from torch import Tensor
import heapq #堆处理模块
import time

# from torch_geometric
def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)

def csr_to_dense(src: Tensor, indptr: Optional[Tensor] = None,
                 fill_value: float = 0., max_degree: Optional[int] = None,
                 batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """
    Convert a CSR list to a neighbor list. Based on PyG to_dense_batch.
    """
    # indptr -> batch
    if indptr is None:
        batch = None
    else:
        ranges = torch.arange(indptr.size(0) - 1, device=indptr.device)
        diffs = torch.diff(indptr)
        batch = torch.repeat_interleave(ranges, diffs)
    
    if batch is None and max_degree is None:
        mask = torch.ones(1, src.size(0), dtype=torch.bool, device=src.device)
        return src.unsqueeze(0), mask

    if batch is None:
        batch = src.new_zeros(src.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter(batch.new_ones(src.size(0)), batch, dim=0,
                        dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    filter_nodes = False
    if max_degree is None:
        max_degree = int(num_nodes.max())
    elif num_nodes.max() > max_degree:
        filter_nodes = True

    tmp = torch.arange(batch.size(0), device=src.device) - cum_nodes[batch]
    idx = tmp + (batch * max_degree)
    if filter_nodes:
        mask = tmp < max_degree
        src, idx = src[mask], idx[mask]

    size = [batch_size * max_degree] + list(src.size())[1:]
    out = src.new_full(size, fill_value)
    out[idx] = src
    out = out.view([batch_size, max_degree] + list(src.size())[1:])

    mask = torch.zeros(batch_size * max_degree, dtype=torch.bool,
                       device=src.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_degree)

    return out, mask

def batched_csr_selection(starts, ends): #返回所有顶点的邻居节点的csr存储格式
    """
    Given start indices and end indices, give its CSR selection tensor and pointer.
    Example:
    starts [0, 2, 5, 18]
    ends   [2, 5, 9, 21]
    
    returns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20]
    ptr: [0, 2, 5, 9, 12]
    """
    device = starts.device
    sizes = ends - starts
    begin_idx = sizes.cumsum(0) #使用sizes[:-1].cumsum(0)和torch.cat会更快嘛？？？？
    ptr = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), begin_idx])
    begin_idx = begin_idx.roll(1)  
    begin_idx[0] = 0
    result = torch.arange(sizes.sum(), device=device) + (starts - begin_idx).repeat_interleave(sizes)
    return result, ptr

def batched_csr_selection_opt(starts, ends): #返回所有顶点的邻居节点的csr存储格式
    sizes = ends - starts
    # print("sizes type:", sizes.dtype)
    begin_idx = sizes.cumsum(0).to(torch.int32) 
    # print("begin_idx", begin_idx)
    # print("begin_idx type:", begin_idx.dtype)
    device = starts.device
    ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), begin_idx])
    begin_idx = ptr[:-1]
    # print("ptr[-1]", ptr[-1])
    # print("starts - begin_idx", starts - begin_idx)
    # print("sizes", sizes)
    result = torch.arange(ptr[-1], device=device) + (starts - begin_idx).repeat_interleave(sizes)
    return result, ptr

def batched_csr_selection_opt2(starts, ends):
    device = starts.device
    sizes = ends - starts
    # print("starts:", starts, starts.shape[0])
    # print("ends:", ends)
    # print("sizes:", sizes)
    begin_idx = sizes.cumsum(0).to(torch.int32)
    # print("begin_idx:", begin_idx)
    ptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), begin_idx])
    # print("ptr", ptr)
    result = torch.ones(begin_idx[-1]+1, dtype=torch.int32, device=device)
    # print("result.shape:", result.shape[0])
    # print(torch.max(ptr[:-1]))
    # print(ptr[-2:])
    result[ptr[:-1]] = starts  #如果最后几个节点都是空就容易出错
    # print("result:", result)
    # correcting start indices for initial values
    result[ptr[1:-1]] += 1-(starts[:-1] + sizes[:-1])
    result = result[:-1]
    return result.cumsum(0).to(torch.int32), ptr

    # arr_len = torch.sum(count)
    # print("arr_len: ", arr_len)

    # # building reset indices
    # ri=torch.zeros(torch.numel(count),dtype=count.dtype,device=count.device)

    # ri[1:]=torch.cumsum(count,dim=0)[:-1]
    # #building incremental indices
    # incr=torch.ones(arr_len,dtype=count.dtype,device=count.device)
    # incr[ri]=start
    # # correcting start indices for initial values
    # incr[ri[1:]]+=1-(start[:-1]+count[:-1])


def batched_adj_selection(starts, ends, mask_value=-1): #返回邻居节点，不足的地方用-1去补齐成长度一致
    """
    For example:
    starts [0, 2, 5, 18]
    ends   [2, 5, 9, 21]
    
    returns (assuming mask_value=-1)
    [[ 0,  1, -1, -1],
     [ 2,  3,  4, -1],
     [ 5,  6,  7,  8],
     [18, 19, 20, -1]],
    and the according mask
    """
    device = starts.device
    sizes = ends - starts
    begin_idx = sizes.cumsum(0)
    max_size = torch.max(sizes)
    result = torch.ones((starts.size(0) * max_size,), dtype=torch.int64, device=device) * mask_value
    begin_idx = begin_idx.roll(1)
    begin_idx[0] = 0
    ranges = torch.arange(sizes.sum(), device=device)
    value = ranges - (begin_idx - starts).repeat_interleave(sizes)
    row_starts = torch.arange(starts.size(0), device=device) * max_size
    idx = ranges + (row_starts - begin_idx).repeat_interleave(sizes)
    result[idx] = value
    result = result.view((starts.size(0), max_size))
    mask = (result != mask_value)
    return result, mask

def divide_equally(data: torch.Tensor, partition_size):
    """
    Partition data into `partition_size` groups, and keep the sum of each group as near as possible.
    Return a list of tensors, which includes the indices of the partitioned data.
    eg: 将元素 8 6 5 4 3 2 1 分为三组，这些元素将依次分到 1 2 3 | 3 2 1 | 1组
    """
    t1 = time.time()
    sorted, indices = torch.sort(data, descending=True) #将数据进行降序排列
    partition_size = min(partition_size, data.size(0)) #数据划分的块数不能大于数据0维的维度
    heap = [(0, idx) for idx in range(partition_size)] #
    heapq.heapify(heap) #以线性时间将一个列表转化为小根堆，实现优先队列
    results = [[] for _ in range(partition_size)] #生成partition_size个空列表
    value_results = [[] for _ in range(partition_size)] 
    data_idx = 0
    while data_idx < data.size(0): #如果元素没有分完
        set_sum, idx = heapq.heappop(heap) #弹出最小的元素
        print(set_sum, idx)
        results[idx].append(indices[data_idx])
        value_results[idx].append(sorted[data_idx])
        set_sum += sorted[data_idx]
        print("set_sum", set_sum)
        heapq.heappush(heap, (set_sum, idx))
        data_idx += 1
    t2 = time.time()
    print(f"divide_equally: {t2 - t1}")
    return [torch.tensor(result, device=data.device) for result in results], \
        [torch.tensor(result, device=data.device) for result in value_results]

def divide_equally_new(data: torch.Tensor, partition_size):
    """
    Partition data into `partition_size` groups, and keep the sum of each group as near as possible.
    Return a list of tensors, which includes the indices of the partitioned data.
    """
    t1 = time.time()
    sorted, indices = torch.sort(data, descending=True) #将数据进行降序排列
    partition_size = min(partition_size, data.size(0)) #数据划分的块数不能大于数据0维的维度
    results = [[] for _ in range(partition_size)] #生成partition_size个空列表
    value_results = [[] for _ in range(partition_size)] 
    for data_idx in range(data.size(0)):
        idx = data_idx % partition_size
        results[idx].append(indices[data_idx])
        value_results[idx].append(sorted[data_idx])
    t2 = time.time()
    print(f"divide_equally: {t2 - t1}")
    return [torch.tensor(result, device=data.device) for result in results], \
        [torch.tensor(result, device=data.device) for result in value_results]

if __name__ == '__main__':
    # src = torch.tensor([0,1,2,3,4,5])
    # indptr = torch.tensor([0,2,3,6])
    # print(csr_to_dense(src, indptr))
    # print([[]] * 2)
    # print([[] for _ in range(2)])
    tens = torch.tensor([0,1,2,3,4,5,6,7,8])
    size = 2
    r1, r2 = divide_equally(tens, size)
    print("r1", r1, "r2", r2)
    r3, r4 = divide_equally_new(tens, size)
    print("r3", r3,"r4", r4)
    