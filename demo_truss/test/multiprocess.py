# import torch
# import torch.multiprocessing as mp

# def worker(gpu_id, task_id):
#     # 将当前进程绑定到指定的GPU
#     torch.cuda.set_device(gpu_id)
#     print(f"Task {task_id} is running on GPU {gpu_id}")

#     # 在不同的GPU上执行不同的任务
#     if task_id == 0:
#         # Task 0: 在 GPU 上进行矩阵乘法
#         a = torch.rand(1000, 1000).cuda()
#         b = torch.rand(1000, 1000).cuda()
#         c = torch.matmul(a, b)
#         print("c", c)
#         print(f"Task {task_id} completed on GPU {gpu_id}")
#     elif task_id == 1:
#         # Task 1: 在 GPU 上进行矩阵加法
#         x = torch.rand(1000, 1000).cuda()
#         y = torch.rand(1000, 1000).cuda()
#         z = x + y
#         print("z", z)
#         print(f"Task {task_id} completed on GPU {gpu_id}")
#     else:
#         print(f"Task {task_id} has nothing to do on GPU {gpu_id}")

# if __name__ == "__main__":
#     num_gpus = torch.cuda.device_count()
#     processes = []

#     # 启动多进程，每个进程绑定一个GPU，并执行不同的任务
#     for gpu_id in range(num_gpus):
#         task_id = gpu_id
#         p = mp.Process(target=worker, args=(gpu_id, task_id))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# import os
# import psutil
# import torch
# from torch.multiprocessing import Process, Manager

# def tensor_operation(rank, shared_list):
#     pid = os.getpid()
#     num_cores = psutil.cpu_count()
#     all_cores = list(range(num_cores))
#     p = psutil.Process(pid)
#     p.cpu_affinity(all_cores)
#     # Your tensor operation here
#     support_tensor = torch.randn(10000, 10000)
#     support_count = (support_tensor > 0).sum().item()
#     shared_list.append((rank, support_count))
#     print(f"Process {rank} finished with support count: {support_count}")

# if __name__ == '__main__':
#     manager = Manager()
#     shared_list = manager.list()
#     processes = []
#     num_processes = 4

#     for rank in range(num_processes):
#         p = Process(target=tensor_operation, args=(rank, shared_list))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     print("Results:")
#     for rank, support_count in shared_list:
#         print(f"Process {rank}: support count = {support_count}")



class mutilGPU_ktruss(Process):
    def __init__(self, rank, size, row_ptr,columns,**kwargs):
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
        print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
        logging.info('num edges:{}.'.format(self.columns.shape[0]))



        
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
    
    print("------------------------------------------------")
    print('loading graph...', end=' ', flush=True) 
    graph = torch.load(args.output)
    print('loading Done!')
    graph.pin_memory()
    print("args",args)
    """if args.cuda:
        graph.to('cuda')
        print('use cuda')"""
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))

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
    
