"""
A Graph type implemented with CSRC (CSR + CSC).
为什么不先判断一下数据是行稀疏还是列稀疏，然后再选择合适的存储方式呢？既生成CSR存储数据，又生成CSC存储数据，不会更费时间和存储容量吗？
"""
from .Graph_Truss import Graph
import torch
import numpy as np

class CSRCOO(Graph):
    """
    CSR + CSC implementation of Graph. Efficient access to out_nbrs and in_nbrs. Assume the graph is directed. (otherwise use CSRGraph). Provides a mapping from CSC row indices to CSR column indices.
    """
    def __init__(self,
                 columns: torch.Tensor=None,
                 row_ptr: torch.Tensor=None,
                 rows: torch.Tensor=None
                ):
        """
        Initialize a CSRCOO object with according datatypes (tensors).
        
        :param Tensor columns: out-neighbors of vertex (arranged in order) (for CSR)
        :param Tensor row_ptr: pointers of each vertex for val and col_ind (for CSR)
        :param Tensor rows: in-neighbors of vertex
        (arranged in order) (for CSC)
        :param Tensor column_ptr: pointers of each vertex for val and row_ind (for CSC)
        :param Tensor shuffle_ptr: pointers from CSC rows to CSR columns.
        #未来加一个csc flag 标记是否使用csc这个数据格式
        """
        super().__init__(directed=True)
        self.rows = rows
        self.columns = columns
        self.row_ptr = row_ptr

                    
    @property  #因为整个计算过程不peeling点，所以以后将顶点属性去掉
    def num_vertices(self):
        """number of vertices."""
        if hasattr(self.row_ptr, 'shape'):
            return self.row_ptr.shape[0] - 1
        else:
            return 0
    
    @property
    def num_edges(self):
        if hasattr(self.columns, 'shape'):
            return self.columns.shape[0]
        else:
            return 0
    
    def out_degree(self, vertices):
        assert torch.all(vertices < self.num_vertices)
        return self.out_degrees[vertices]
    
    def in_degree(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def out_nbrs(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def out_nbrs_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def all_out_nbrs_csr(self):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def in_nbrs(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def in_nbrs_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def all_in_nbrs_csr(self):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def out_edges(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def out_edges_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def all_out_edges_csr(self):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def in_edges(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def in_edges_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def all_in_edges_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    @property
    def device(self):
        col_ind_dev = self.columns.device
        row_ind_dev = self.row_ptr.device
        assert col_ind_dev == row_ind_dev, "Graph is not on the same device."
        return col_ind_dev
    
    def to(self, *args, **kwargs):#*args 是一个可变数量的参数列表；**kwargs 是一个可变数量的参数字典
        # if self.vertices_t is not None:
        #     self.vertices_t = self.vertices_t.to(*args, **kwargs)
        # if self.edges_t is not None:
        #     self.edges_t = self.edges_t.to(*args, **kwargs)
        self.columns = self.columns.to(*args, **kwargs)
        self.row_ptr = self.row_ptr.to(*args, **kwargs)
        self.rows = self.rows.to(*args, **kwargs)
        # self.column_ptr = self.column_ptr.to(*args, **kwargs)
        # self.shuffle_ptr = self.shuffle_ptr.to(*args, **kwargs)
        
    def pin_memory(self):
        self.columns = self.columns.pin_memory()
        self.row_ptr = self.row_ptr.pin_memory()
        self.rows = self.rows.pin_memory()
        # self.column_ptr = self.column_ptr.pin_memory()
        # self.shuffle_ptr = self.shuffle_ptr.pin_memory()
        
    def subgraph(self, vertices: torch.Tensor):
        """
        Get a subgraph induced by the given vertices.
        """
        # First convert to edge list, then convert to CSRCOO
        raise NotImplementedError('Not implemented for CSRCOO.')
    
    def csr_subgraph(self, vertices: torch.Tensor):
        raise NotImplementedError('Not implemented for CSRCOO.')
        
    @staticmethod  #处理有向图？
    def edge_list_to_Graph(edge_starts, edge_ends, vertices=None):
        print("edge_starts:", edge_starts)
        print("edge_ends", edge_ends)
        # get vertex to index mapping
        #默认顶点属性是按顺序有所有顶点的属性的，列表第一个值就是顶点1的属性，第二个值是顶点2的属性
        vertex_to_index = {} #创建一个空的字典，用于存储顶点到索引的映射关系。
        for vertex, index in zip(vertices, range(len(vertices))): #使用for index, vertexin in enumerate(vertices): 更好吧;同时迭代两个序列：vertices是顶点列表，而range(len(vertices))是整数序列
            vertex_to_index[vertex] = index
        edge_starts = torch.tensor([vertex_to_index[i] for i in edge_starts], dtype= torch.int32)
        edge_ends = torch.tensor([vertex_to_index[i] for i in edge_ends], dtype= torch.int32)
        print("edge_starts:", edge_starts)
        print("edge_ends", edge_ends)
        #顶点重新编号后可能出现edge_start>edge_end的情况，这里需要重排序
        mask = edge_starts > edge_ends
        edge_starts[mask], edge_ends[mask] = edge_ends[mask], edge_starts[mask]
        print("edge_starts:", edge_starts)
        print("edge_ends", edge_ends)
        # Counduct counter sort
        del mask
        row_ptr, pos_sources = CSRCOO.counter_sort(edge_starts, len(vertices)) #csr存储
        print('row_ptr:{}, pos_sources:{}'.format(row_ptr, pos_sources)) #row_ptr是对所有顶点进行索引后的csr格式的行偏移量，pos_sources是起点按照升序排列的索引
        #上面的处理每个分段columns内部是无序的，所以增加下面几行
        for i in range(row_ptr.shape[0]-1):
            temp_pos = pos_sources[row_ptr[i]:row_ptr[i+1]]
            pos = torch.argsort(edge_ends[temp_pos])
            pos_sources[row_ptr[i]:row_ptr[i+1]] = temp_pos[pos]
        # print("pos_sources:", pos_sources)
        columns = edge_ends[pos_sources]  #记录每个元素的列数
        rows = edge_starts[pos_sources] #
        return CSRCOO(
            columns=columns, #csr的列值
            row_ptr = row_ptr.to(torch.int32),  #csr的行偏移量
            rows = rows #csc的行值
        ), vertex_to_index
    
    @staticmethod
    def read_graph(f, split=None,  directed=True):
        edge_starts, edge_ends, vertices = Graph.read_edgelist(f, split, directed)
        return CSRCOO.edge_list_to_Graph(edge_starts, edge_ends, vertices=vertices)
    
    @staticmethod
    def counter_sort(tensor: torch.Tensor, num_vertices):
        """
        Implements counter sort. counts[i] is the number of elements in tensor that are less than or equal to i. pos[i] is the position of the i-th smallest element in tensor.
        """
        counts = torch.cumsum(torch.bincount(tensor, minlength=num_vertices), dim=-1,  dtype=torch.int32)
        counts = torch.cat((torch.tensor([0]), counts))
        pos = torch.argsort(tensor)#返回一个张量中元素按升序排列的索引
        return counts, pos
