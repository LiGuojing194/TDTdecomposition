"""
Abstract interface for the Graph data type.
"""

import abc
import numpy as np
import torch

class Graph(abc.ABC):
    def __init__(self, directed=False):
        self.directed = directed     #默认设置为无向图
        self.vertices_t, self.edges_t = None, None   #顶点和边先设置为空
        
    @property #将一个方法伪装成属性，被修饰的特性方法，内部可以实现处理逻辑，但对外提供统一的调用方式
    @abc.abstractmethod #抽象方法，含abstractmethod方法的类不能实例化，继承了含abstractmethod方法的子类必须复写所有abstractmethod装饰的方法，未被装饰的可以不重写
    def num_vertices(self):
        pass
    
    @property
    def vertices(self):
        if self.vertices_t is None:
            self.vertices_t = torch.arange(self.num_vertices, device=self.device)
        return self.vertices_t
    
    @property
    @abc.abstractmethod
    def num_edges(self):
        pass
    
    @property
    def edges(self):
        if self.edges_t is None:
            self.edges_t = torch.arange(self.num_edges, device=self.device)
        return self.edges_t
    
    @abc.abstractmethod
    def out_degree(self, vertices): #出度
        pass
    
    @abc.abstractmethod
    def in_degree(self, vertices): #入度
        pass
    
    def all_degree(self, vertices):
        if not self.directed:
            return self.out_degree(vertices)
        return self.out_degree(vertices) + self.in_degree(vertices)
    
    @abc.abstractmethod
    def out_nbrs(self, vertices):
        pass
        #要返回out_n, out_n_mask这两个值
    
    @abc.abstractmethod
    def in_nbrs(self, vertices):
        pass
        #要返回in_n, in_n_mask这两个值
    
    def all_nbrs(self, vertices):
        if not self.directed:
            return self.out_nbrs(vertices)
        out_n, out_n_mask = self.out_nbrs(vertices) #这两句什么意思？ 这两个函数都pass了，还没被具体化。
        in_n, in_n_mask = self.in_nbrs(vertices) #怎么有两个返回值，怎么返回的？要看继承时具体化的实体类中的这个方法
        return torch.cat((out_n, in_n), dim=1), torch.cat((out_n_mask, in_n_mask), dim=1) #torch.cat(, dim=1) 竖着拼
    
    @abc.abstractmethod
    def out_nbrs_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_out_nbrs_csr(self):
        pass
    
    @abc.abstractmethod
    def in_nbrs_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_in_nbrs_csr(self):
        pass
    #return out_n, out_n_ptr

    def all_nbrs_csr(self, vertices):
        if not self.directed:
            return self.out_nbrs_csr(vertices)
        #以下是对有向图的处理
        out_n, out_n_ptr = self.out_nbrs_csr(vertices)
        in_n, in_n_ptr = self.in_nbrs_csr(vertices)
        ptr = out_n_ptr + in_n_ptr
        # torch.device代表将torch.Tensor分配到的设备的对象。
        # torch.device包含一个设备类型（'cpu'或'cuda'设备类型）和可选的设备的序号
        # torch.device('cuda:0') 或 torch.device('cpu', 0)
        # cuda1 = torch.device('cuda:1')
        # torch.randn((2,3), device=cuda1)
        nbrs = torch.zeros((out_n.shape[0] + in_n.shape[0]), dtype=out_n.dtype, device=out_n.device)
        curr_beg = 0
        for i in range(1, len(ptr)+1):
            curr_end = curr_beg + out_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[out_n_ptr[i-1]:out_n_ptr[i]]
            curr_beg = curr_end
            curr_end = curr_beg + in_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[in_n_ptr[i-1]:in_n_ptr[i]]
            curr_beg = curr_end
        return nbrs, ptr
    
    @abc.abstractmethod
    def out_edges(self, vertices):
        pass
    
    @abc.abstractmethod
    def in_edges(self, vertices):
        pass
    
    def all_edges(self, vertices):
        if not self.directed:
            return self.out_edges(vertices)
        out_e, out_e_mask = self.out_edges(vertices)  #同理？？？？这个函数是抽象函数，在继承这个类的时候被具体化
        in_e, in_e_mask = self.in_edges(vertices)      #在继承这个类的时候被具体化
        return torch.cat((out_e, in_e), dim=1), torch.cat((out_e_mask, in_e_mask), dim=1)
    
    @abc.abstractmethod
    def out_edges_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_out_edges_csr(self):
        pass
    
    @abc.abstractmethod
    def in_edges_csr(self, vertices):
        pass
    
    @abc.abstractmethod
    def all_in_edges_csr(self):
        pass
    
    def all_edges_csr(self, vertices):
        if not self.directed:
            return self.out_edges_csr(vertices)
        out_n, out_n_ptr = self.out_edges_csr(vertices)
        in_n, in_n_ptr = self.in_edges_csr(vertices)
        ptr = out_n_ptr + in_n_ptr
        nbrs = torch.zeros((out_n.shape[0] + in_n.shape[0]), dtype=out_n.dtype, device=out_n.device)
        curr_beg = 0
        for i in range(1, len(ptr)+1):
            curr_end = curr_beg + out_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[out_n_ptr[i-1]:out_n_ptr[i]]
            curr_beg = curr_end
            curr_end = curr_beg + in_n_ptr[i]
            nbrs[curr_beg:curr_end] = out_n[in_n_ptr[i-1]:in_n_ptr[i]]
            curr_beg = curr_end
        return nbrs, ptr
    
    @abc.abstractmethod
    def device(self):
        pass
    
    @abc.abstractmethod
    def to(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def pin_memory(self):
        pass
    
    @abc.abstractmethod
    def subgraph(self, vertices):
        """
        Induced subgraph from vertices.
        """
        pass
    
    @abc.abstractmethod
    def csr_subgraph(self, vertices: torch.Tensor):
        pass
    
    def read_edgelist(f, split, directed):
        """
        Read edge-list from a file. Allow one value for each edge.
        
        :param f: file to read from
        :param str split: split string, such as spaces or tabs.
        :return: edge_starts, edge_ends, vertices, edge_data (a list of np.arrays, each is a column)
        这个函数去除了重边，只保留1->2这种边； 并对顶点进行了度排序
        """
        print('-------- {} ------------'.format(f))
        array = np.loadtxt(f, dtype=np.int32)
        array = array[(array[:, 0]!=array[:, 1]), :]  #删除环  *****
        # array[:, [0, 1]] = np.sort(array[:, [0, 1]]) #对数据排过序了，start<end;  后面edge_list_to_Graph中有对重边进行删除
        # print("array", array)
        #去除重边和环：#unique预处理时间太长了
        # if not directed:
        #     # _, idx = np.unique(np.sort(array[:, [0, 1]]), return_index=True, axis=0)
        #     #去除环 和 1->2,2->1这种重边：
        # else:#去了环，但是重边没处理，这样的话，使用CSRC时会出问题
        #     # _, idx = np.unique(array[:, [0, 1]], return_index=True, axis=0)
        #去除重边，为了统计每个顶点的度， #删除1->2,2->1这种重边，，，对于真正意义上的无向图，需要对此进行修改
        #改成start < end
        # array[:, [0, 1]] = 
        _, idx = np.unique(np.sort(array[:, [0, 1]]), return_index=True, axis=0)   #*******
        edge_starts = array[idx, 0]  #******
        edge_ends = array[idx, 1] #******
        # edge_starts = array[:, 0]
        # edge_ends = array[:, 1]
        # data = array[idx, 2:].T 
        # print("edge_starts:{}".format(edge_starts))
        # print("edge_ends:{}".format(edge_ends))
        #进行顶点度排序
        vertices, counts = np.unique(np.concatenate((edge_starts, edge_ends)), return_counts=True)#np.sort返回一个新的数组，其中的元素按升序排列（从小到大）
        idx = np.argsort(counts)
        vertices = vertices[idx]
        # vertices = np.unique(np.concatenate((edge_starts, edge_ends)))
        return edge_starts, edge_ends, vertices
