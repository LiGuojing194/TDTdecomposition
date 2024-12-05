"""
A Graph type implemented with CSC (compressed sparse column) type.
"""

import torch
from .CSRGraph import CSRGraph
from .Graph import Graph
import numpy as np

class CSCGraph(Graph):
    """
    CSC (compressed sparse column) implementation of graphs. Efficient access to in_nbrs. This is implemented as an adapter to CSRGraph.
    """
    def __init__(self,
                 rows: torch.Tensor=None,
                 column_ptr: torch.Tensor=None,
                 csr: CSRGraph=None,
                 directed=False,
                 vertex_attrs_list=[],
                 vertex_attrs_tensor: torch.Tensor=None,
                 vertex_attrs_mask: torch.Tensor=None,
                 edge_attrs_list=[],
                 edge_attrs_tensor: torch.Tensor=None,
                 edge_attrs_mask: torch.Tensor=None):
        """
        Initialize a CSCGraph object with according datatypes (tensors).
        
        :param Tensor rows: in-neighbors of vertex (arranged in order)
        :param Tensor column_ptr: pointers of each vertex for val and row_ind
        :param bool directed: whether the graph is directed
        :param list vertex_attrs_list: list of vertex attributes names
        :param Tensor vertex_attrs_tensor: tensor of vertex attributes that stores data
        :param Tensor vertex_attrs_mask: mask of vertex attributes
        :param list edge_attrs_list: list of edge attributes names
        :param Tensor edge_attrs_tensor: tensor of edge attributes that stores data
        :param Tensor edge_attrs_mask: mask of edge attributes
        :return: None
        """
        super().__init__(directed)
        if csr is not None:
            self.csr = csr
        else:
            self.csr = CSRGraph(columns=rows, 
                                row_ptr=column_ptr, 
                                directed=directed,
                                vertex_attrs_list=vertex_attrs_list,
                                vertex_attrs_tensor=vertex_attrs_tensor,
                                vertex_attrs_mask=vertex_attrs_mask,
                                edge_attrs_list=edge_attrs_list,
                                edge_attrs_tensor=edge_attrs_tensor,
                                edge_attrs_mask=edge_attrs_mask)

     #以下@property修饰的方法用于设置用户只读而不能修改的属性   
    @property
    def num_vertices(self):
        return self.csr.num_vertices
    
    @property
    def num_edges(self):
        return self.csr.num_edges
    
    def out_degree(self, vertices):
        return self.csr.in_degree(vertices)
    
    def in_degree(self, vertices):
        return self.csr.out_degree(vertices)
    
    def out_nbrs(self, vertices):
        raise NotImplementedError('Not implemented for CSCGraph.')
    
    def out_nbrs_csr(self, vertices):
        return self.csr.in_nbrs_csr(vertices)
    
    def all_out_nbrs_csr(self):
        return self.csr.all_in_nbrs_csr()
    
    def in_nbrs(self, vertices):
        return self.csr.out_nbrs(vertices)
    
    def in_nbrs_csr(self, vertices):
        return self.csr.out_nbrs_csr(vertices)
    
    def all_in_nbrs_csr(self):
        return self.csr.all_out_nbrs_csr()
    
    def out_edges(self, vertices):
        raise NotImplementedError('Not implemented for CSCGraph.')
    
    def out_edges_csr(self, vertices):
        return self.csr.in_edges_csr(vertices)
    
    def all_out_edges_csr(self):
        return self.csr.all_in_edges_csr()
    
    def in_edges(self, vertices):
        return self.csr.out_edges(vertices)
    
    def in_edges_csr(self, vertices):
        return self.csr.out_edges_csr(vertices)
    
    def all_in_edges_csr(self):
        return self.csr.all_out_edges_csr()
    
    @property
    def device(self):
        return self.csr.device
    
    def to(self, *args, **kwargs):
        self.csr.to(*args, **kwargs)
        
    def pin_memory(self):
        self.csr.pin_memory()
        
    def subgraph(self, vertices):
        csr, n_to_o, _ = self.csr.subgraph(vertices)
        return CSCGraph(csr=csr, directed=csr.directed), n_to_o
    
    def csr_subgraph(self, vertices: torch.Tensor):
        csr, indices = self.csr.csr_subgraph(vertices)
        return CSCGraph(csr=csr, directed=csr.directed), indices
    
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
        
    @staticmethod
    def edge_list_to_Graph(edge_starts, edge_ends, directed=False, vertices=None, edge_attrs=None, edge_attrs_list=[], vertex_attrs=None, vertex_attrs_list=[]):
        csr, vtid, tensors = CSRGraph.edge_list_to_Graph(edge_ends, edge_starts, directed=directed, vertices=vertices, edge_attrs=edge_attrs,
                                                         edge_attrs_list=edge_attrs_list, vertex_attrs=vertex_attrs, vertex_attrs_list=vertex_attrs_list)
        return CSCGraph(csr=csr, directed=directed), vtid, tensors
        
    @staticmethod
    def read_graph(f, split=' ', directed=False):
        edge_starts, edge_ends, vertices, data = Graph.read_edgelist(f, split)
        return CSCGraph.edge_list_to_Graph(edge_starts, edge_ends, directed=directed, vertices=vertices) 