"""
A Graph type implemented with CSRC (CSR + CSC).
为什么不先判断一下数据是行稀疏还是列稀疏，然后再选择合适的存储方式呢？既生成CSR存储数据，又生成CSC存储数据，不会更费时间和存储容量吗？
"""
from .Graph import Graph
from .CSRGraph import CSRGraph
from .CSCGraph import CSCGraph
import torch
import numpy as np
import timeit
import time

class CSRCGraph(Graph):
    """
    CSR + CSC implementation of Graph. Efficient access to out_nbrs and in_nbrs. Assume the graph is directed. (otherwise use CSRGraph). Provides a mapping from CSC row indices to CSR column indices.
    """
    def __init__(self,
                 shuffle_ptr: torch.Tensor,
                 columns: torch.Tensor=None,
                 row_ptr: torch.Tensor=None,
                 rows: torch.Tensor=None,
                 column_ptr: torch.Tensor=None,
                 csr: CSRGraph=None,
                 csc: CSCGraph=None,
                 vertex_attrs_list=[],
                 vertex_attrs_tensor: torch.Tensor=None,
                 vertex_attrs_mask: torch.Tensor=None,
                 edge_attrs_list=[],
                 edge_attrs_tensor: torch.Tensor=None,
                 edge_attrs_mask: torch.Tensor=None):
        """
        Initialize a CSRCGraph object with according datatypes (tensors).
        
        :param Tensor columns: out-neighbors of vertex (arranged in order) (for CSR)
        :param Tensor row_ptr: pointers of each vertex for val and col_ind (for CSR)
        :param Tensor rows: in-neighbors of vertex
        (arranged in order) (for CSC)
        :param Tensor column_ptr: pointers of each vertex for val and row_ind (for CSC)
        :param Tensor shuffle_ptr: pointers from CSC rows to CSR columns.
        :param list vertex_attrs_list: list of vertex attributes names
        :param Tensor vertex_attrs_tensor: tensor of vertex attributes that stores data
        :param Tensor vertex_attrs_mask: mask of vertex attributes
        :param list edge_attrs_list: list of edge attributes names
        :param Tensor edge_attrs_tensor: tensor of edge attributes that stores data
        :param Tensor edge_attrs_mask: mask of edge attributes
        """
        super().__init__(directed=True)
        if csr is not None:
            self.csr = csr
        else:
            self.csr = CSRGraph(columns=columns, row_ptr=row_ptr, directed=True,
                                vertex_attrs_list=vertex_attrs_list, vertex_attrs_tensor=vertex_attrs_tensor, vertex_attrs_mask=vertex_attrs_mask,
                                edge_attrs_list=edge_attrs_list, edge_attrs_tensor=edge_attrs_tensor, edge_attrs_mask=edge_attrs_mask)
        if csc is not None:
            self.csc = csc
        else:
            self.csc = CSCGraph(rows=rows, column_ptr=column_ptr, directed=True,vertex_attrs_list=vertex_attrs_list, vertex_attrs_tensor=vertex_attrs_tensor, vertex_attrs_mask=vertex_attrs_mask,
                                edge_attrs_list=edge_attrs_list, edge_attrs_tensor=edge_attrs_tensor, edge_attrs_mask=edge_attrs_mask)
        self.shuffle_ptr = shuffle_ptr
                    
    @property
    def num_vertices(self):
        return self.csr.num_vertices
    
    @property
    def num_edges(self):
        return self.csr.num_edges
    
    def out_degree(self, vertices):
        return self.csr.out_degree(vertices)
    
    def in_degree(self, vertices):
        return self.csc.in_degree(vertices)
    
    def out_nbrs(self, vertices):
        return self.csr.out_nbrs(vertices)
    
    def out_nbrs_csr(self, vertices):
        return self.csr.out_nbrs_csr(vertices)
    
    def all_out_nbrs_csr(self):
        return self.csr.all_out_nbrs_csr()
    
    def in_nbrs(self, vertices):
        return self.csc.in_nbrs(vertices)
    
    def in_nbrs_csr(self, vertices):
        return self.csc.in_nbrs_csr(vertices)
    
    def all_in_nbrs_csr(self):
        return self.csc.all_in_nbrs_csr()
    
    def out_edges(self, vertices):
        return self.csr.out_edges(vertices)
    
    def out_edges_csr(self, vertices):
        return self.csr.out_edges_csr(vertices)
    
    def all_out_edges_csr(self):
        return self.csr.all_out_edges_csr()
    
    def in_edges(self, vertices):
        csc_in_edges, csc_masks = self.csc.in_edges(vertices)
        # in_edges = self.shuffle_ptr[csc_in_edges]
        in_edges = torch.where(csc_masks, self.shuffle_ptr[csc_in_edges], torch.ones_like(csc_in_edges) * -1)
        return in_edges, csc_masks
    
    def in_edges_csr(self, vertices):
        csc_in_edges, ptr = self.csc.in_edges_csr(vertices)
        in_edges = self.shuffle_ptr[csc_in_edges]
        return in_edges, ptr
    
    def all_in_edges_csr(self, vertices):
        return self.shuffle_ptr, self.csc.csr.row_ptr
    
    @property
    def device(self):
        return self.csr.device
    
    def to(self, *args, **kwargs):#*args 是一个可变数量的参数列表；**kwargs 是一个可变数量的参数字典
        if self.vertices_t is not None:
            self.vertices_t = self.vertices_t.to(*args, **kwargs)
        if self.edges_t is not None:
            self.edges_t = self.edges_t.to(*args, **kwargs)
        self.csr.to(*args, **kwargs)
        self.csc.to(*args, **kwargs)
        self.shuffle_ptr = self.shuffle_ptr.to(*args, **kwargs)
        
    def pin_memory(self):
        self.csr.pin_memory()
        self.csc.pin_memory()
        self.shuffle_ptr = self.shuffle_ptr.pin_memory()
        
    def subgraph(self, vertices: torch.Tensor):
        """
        Get a subgraph induced by the given vertices.
        """
        # First convert to edge list, then convert to CSRCGraph
        edge_starts = []
        edge_ends = []
        for v in vertices:
            for nbr in self.out_nbrs(v):
                if nbr in vertices:
                    edge_starts.append(v)
                    edge_ends.append(nbr)
        return CSRCGraph.edge_list_to_Graph(edge_starts, edge_ends)
    
    def csr_subgraph(self, vertices: torch.Tensor):
        new_csr, indices_csr = self.csr.csr_subgraph(vertices)
        new_csc, indices_csc = self.csc.csr_subgraph(vertices)
        new_shuffle_ptr = self.shuffle_ptr[indices_csc]   #这一行为什么要这样设置呢？
        return CSRCGraph(csr=new_csr, csc=new_csc, shuffle_ptr=new_shuffle_ptr), \
            indices_csr
    
    def get_vertex_attr(self, vertices, attr):
        return self.csr.get_vertex_attr(vertices, attr)
    
    def select_vertex_by_attr(self, attr, cond):
        return self.csr.select_vertex_by_attr(attr, cond)
    
    def set_vertex_attr(self, vertices, attr, value, mask):
        return self.csr.set_vertex_attr(vertices, attr, value, mask)
    
    def get_edge_attr(self, edges, attr):
        return self.csr.get_edge_attr(edges, attr)
    
    def select_edge_by_attr(self, attr, cond):
        return self.csr.select_edge_by_attr(attr, cond)
    
    def set_edge_attr(self, edges, attr, value, mask):
        return self.csr.set_edge_attr(edges, attr, value, mask)
        
    @staticmethod  #处理有向图？
    def edge_list_to_Graph(edge_starts, edge_ends, vertices=None, edge_attrs=None, edge_attrs_list=[], vertex_attrs=None, vertex_attrs_list=[]):
        # get vertex to index mapping
        #默认顶点属性是按顺序有所有顶点的属性的，列表第一个值就是顶点1的属性，第二个值是顶点2的属性
        # vertex_attrs_list=[1]
        # vertex_attrs = [[1, 2, 3, 4, 5, 6, 7]]
        vertex_to_index = {}
        vertex_data_list = [[] for _ in range(len(vertex_attrs_list))]  #先定义一个顶点属性列表空集
        # print('vertex_data_list:{}'.format(vertex_data_list))
        for vertex, index in zip(vertices, range(len(vertices))): #使用for index, vertexin in enumerate(vertices): 更好吧;同时迭代两个序列：vertices是顶点列表，而range(len(vertices))是整数序列
            vertex_to_index[vertex] = index
            if vertex_attrs is not None:
                for data_index, data in enumerate(vertex_attrs):
                    vertex_data_list[data_index].append(data[vertex])#????源代码是vertex_data_list[data_index].append(data[vertex])
        # print('vertex_to_index{}'.format(vertex_to_index))
        # print('vertex_data_list:{}'.format(vertex_data_list))
        # Change edge_starts and edge_ends to indices
        edge_starts = torch.LongTensor([vertex_to_index[i] for i in edge_starts])
        edge_ends = torch.LongTensor([vertex_to_index[i] for i in edge_ends])
        data_tensors = [torch.FloatTensor(i) for i in edge_attrs] #将边属性数据转为一个个张量
        # Counduct counter sort
        row_ptr, pos_sources = CSRCGraph.counter_sort(edge_starts, len(vertices)) #csr存储
        # print('row_ptr:{}, pos_sources:{}'.format(row_ptr, pos_sources)) #row_ptr是对所有顶点进行索引后的csr格式的行偏移量，pos_sources是起点按照升序排列的索引
        #上面的处理每个分段columns内部是无序的，所以增加下面几行
        for i in range(row_ptr.shape[0]-1):
            temp_pos = pos_sources[row_ptr[i]:row_ptr[i+1]]
            pos = torch.argsort(edge_ends[temp_pos])
            pos_sources[row_ptr[i]:row_ptr[i+1]] = row_ptr[i] + pos
        columns = edge_ends[pos_sources]  #记录每个元素的列数
        edge_starts = edge_starts[pos_sources] #
        edge_ends = edge_ends[pos_sources] #将边数据按照起点升序进行排列
        # for i, t in enumerate(data_tensors):#????对边权重也进行重新排序，原始代码是：for t in data_tensors: t = t[pos_sources]，变量t是临时变量，将重新索引（切片）值赋给临时变量，是不会改变原始列表中的元素的
        #     data_tensors[i] = t[pos_sources]
        # print('pre: data_tensors:{},'.format(data_tensors))
        for t in data_tensors:
            t[:] = t[pos_sources]
        # print('after: data_tensors:{},'.format(data_tensors))
        column_ptr, pos_targets = CSRCGraph.counter_sort(edge_ends, len(vertices)) #csc存储结构处理
        rows = edge_starts[pos_targets] #列升序处理后的行索引，？？？？这后面边属性又不按照列升序进行处理？？？？

        if len(data_tensors) != 0: #如果边有权重
            edge_attrs_tensor = torch.stack(data_tensors, dim=0)#将a个1*n的tensor合成一个a*n的tensor
            # print("edge_attrs_tensor:", edge_attrs_tensor)
            edge_attrs_mask = torch.ones(edge_attrs_tensor.shape, dtype=torch.bool) # 创建一个与 edge_attrs_tensor 形状相同的全为 True 的布尔类型张量
        else:
            edge_attrs_tensor, edge_attrs_mask = None, None
        if vertex_attrs is not None: #如果顶点有属性
            vertex_attrs_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in vertex_data_list], dim=0)
            # print('vertex_attrs_tensor:{}'.format(vertex_attrs_tensor))
            vertex_attrs_mask = torch.ones(vertex_attrs_tensor.shape, dtype=torch.bool)
        else:
            vertex_attrs_tensor = None
            vertex_attrs_mask = None
        # print('CSRCGraph Read :columns:{}, row_ptr:{}'.format(columns, row_ptr))
        return CSRCGraph(
            shuffle_ptr=pos_targets, #终点按照升序排列的索引，，pos_sources起点按照升序排列的索引为啥不返回呢？？？
            columns=columns, #csr的列值
            row_ptr=row_ptr,  #csr的行偏移量
            rows=rows, #csc的行值
            column_ptr=column_ptr, #csc的列偏移量
            vertex_attrs_tensor=vertex_attrs_tensor,
            vertex_attrs_list=vertex_attrs_list,
            vertex_attrs_mask=vertex_attrs_mask,
            edge_attrs_tensor=edge_attrs_tensor,
            edge_attrs_list=edge_attrs_list,
            edge_attrs_mask=edge_attrs_mask
        ), vertex_to_index, data_tensors
    
    @staticmethod
    def read_graph(f, split=None,  directed=True):
        edge_starts, edge_ends, vertices, data = Graph.read_edgelist(f, split, directed)
        # print('CSRCGraph.edge_list_to_Graph: edge_starts{}, edge_ends{}, vertices{}, edge_attrs{}'.format(edge_starts, edge_ends, vertices, data))
        return CSRCGraph.edge_list_to_Graph(edge_starts, edge_ends, vertices=vertices, edge_attrs=data)
    
    @staticmethod
    def counter_sort(tensor: torch.Tensor, num_vertices):
        """
        Implements counter sort. counts[i] is the number of elements in tensor that are less than or equal to i. pos[i] is the position of the i-th smallest element in tensor.
        """
        counts = torch.cumsum(torch.bincount(tensor, minlength=num_vertices), dim=-1)
        counts = torch.cat((torch.tensor([0]), counts))
        pos = torch.argsort(tensor)#返回一个张量中元素按升序排列的索引
        return counts, pos
