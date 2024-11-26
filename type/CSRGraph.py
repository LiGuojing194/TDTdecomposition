"""
A Graph type implemented with CSR (compressed sparse row) type.
"""
import torch
import numpy as np
from .Graph import Graph
from ..framework.helper import batched_csr_selection, batched_adj_selection


class CSRGraph(Graph):
    """
    CSR implementation of Graph. Provides efficient access to out_nbrs.
    """
    def __init__(self,
                 columns: torch.Tensor=None, 
                 row_ptr: torch.Tensor=None, 
                 directed=False,
                 vertex_attrs_list=[],
                 vertex_attrs_tensor: torch.Tensor=None,
                 vertex_attrs_mask: torch.Tensor=None,
                 edge_attrs_list=[],
                 edge_attrs_tensor: torch.Tensor=None,
                 edge_attrs_mask: torch.Tensor=None):
        """
        Initialize a CSRGraph object with according datatypes (tensors).
        
        :param Tensor columns: out-neighbors of vertex (arranged in order)
        :param Tensor row_ptr: pointers of each vertex for val and col_ind
        :param bool directed: whether the graph is directed
        :param list vertex_attrs_list: list of vertex attributes names
        :param Tensor vertex_attrs_tensor: tensor of vertex attributes that stores data
        :param Tensor vertex_attrs_mask: mask of vertex attributes
        :param list edge_attrs_list: list of edge attributes names
        :param Tensor edge_attrs_tensor: tensor of edge attributes that stores data
        :param Tensor edge_attrs_mask: mask of edge attributes
        :return: None
        """
        super().__init__(directed=directed)
        self.columns = columns
        self.row_ptr = row_ptr
        self.out_degrees = torch.diff(self.row_ptr)
        # process attributes
        self.vertex_attrs_list = vertex_attrs_list
        self.vertex_attrs_map = {attr: i for i, attr in enumerate(vertex_attrs_list)}
        self.edge_attrs_list = edge_attrs_list
        self.edge_attrs_map = {attr: i for i, attr in enumerate(edge_attrs_list)}
        if vertex_attrs_tensor is not None and vertex_attrs_mask is not None:
            self.vertex_attrs_tensor = vertex_attrs_tensor
            self.vertex_attrs_mask = vertex_attrs_mask
        else:
            self.vertex_attrs_tensor = torch.zeros((self.num_vertices, len(vertex_attrs_list)), dtype=torch.float32)
            self.vertex_attrs_mask = torch.zeros((self.num_vertices, len(vertex_attrs_list)), dtype=torch.bool)
        if edge_attrs_tensor is not None and edge_attrs_mask is not None:
            self.edge_attrs_tensor = edge_attrs_tensor
            self.edge_attrs_mask = edge_attrs_mask
        else:
            self.edge_attrs_tensor = torch.zeros((self.num_edges, len(edge_attrs_list)), dtype=torch.float32)
            self.edge_attrs_mask = torch.zeros((self.num_edges, len(edge_attrs_list)), dtype=torch.bool)
            
    @property
    def num_vertices(self):
        """number of vertices."""
        if hasattr(self.row_ptr, 'shape'):
            return self.row_ptr.shape[0] - 1
        else:
            return 0
        
    @property
    def num_edges(self):
        """number of edges."""
        if hasattr(self.columns, 'shape'):
            return self.columns.shape[0]
        else:
            return 0
    
    def out_degree(self, vertices):
        """
        Get the number of out neighbors. (if undirected, #out_nbrs = #in_nbrs)
        :return: # of out neighbors
        """
        assert torch.all(vertices < self.num_vertices)
        return self.out_degrees[vertices]
    
    def in_degree(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def out_nbrs(self, vertices):
        """
        Get the out neighbors of the vertex.
        :return: out neighbors
        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, mask = batched_adj_selection(starts, ends)
        result = torch.where(mask, self.columns[result], torch.ones_like(result) * -1)
        return result, mask
    
    def out_nbrs_csr(self, vertices):
        # print('vertices: {}'.format(vertices))
        # print('num : {}'.format(self.num_vertices))
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, ptr = batched_csr_selection(starts, ends)
        result = self.columns[result]
        return result, ptr
    
    def all_out_nbrs_csr(self):
        return self.columns, self.row_ptr

    def in_nbrs(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def in_nbrs_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def all_in_nbrs_csr(self):
        raise NotImplementedError('Not implemented for CSRGraph.')
     
    def out_edges(self, vertices):
        """
        Get the out edges of the vertex.
        :return: out edges
        """
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, mask = batched_adj_selection(starts, ends)
        return result, mask
    
    def all_out_edges_csr(self):
        return torch.arange(self.num_edges, device=self.device), self.row_ptr
    
    def out_edges_csr(self, vertices):
        assert torch.all(vertices < self.num_vertices)
        starts = self.row_ptr[vertices]
        ends = self.row_ptr[vertices + 1]
        result, ptr = batched_csr_selection(starts, ends)
        return result, ptr
    
    def in_edges(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def in_edges_csr(self, vertices):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    def all_in_edges_csr(self):
        raise NotImplementedError('Not implemented for CSRGraph.')
    
    @property
    def device(self):
        """
        return the device where the graph resides.
        :return: device
        """
        col_ind_dev = self.columns.device
        row_ind_dev = self.row_ptr.device
        assert col_ind_dev == row_ind_dev, "Graph is not on the same device."
        
        return col_ind_dev
        
    def to(self, *args, **kwargs):
        """
        Move the graph to the specified device.
        
        :return: None
        """
        if self.vertices_t is not None:
            self.vertices_t = self.vertices_t.to(*args, **kwargs)
        if self.edges_t is not None:
            self.edges_t = self.edges_t.to(*args, **kwargs)
        self.columns = self.columns.to(*args, **kwargs)
        self.row_ptr = self.row_ptr.to(*args, **kwargs)
        self.out_degrees = self.out_degrees.to(*args, **kwargs)
        
    def pin_memory(self):  #这是空着没具体实现吧？
        self.columns = self.columns.pin_memory()
        self.row_ptr = self.row_ptr.pin_memory()
    
    def csr_subgraph(self, vertices: torch.Tensor):
        # csr_subgraph yields subgraphs with full neighbors, which is suited for partitioned computing. this is not vertex-induced subgraph.
        sub_degrees = self.out_degrees[vertices]
        sub_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device), sub_degrees.cumsum(0)])
        
        # fetch sub_columns
        # starts, ends = sub_row_ptr[:-1], sub_row_ptr[1:]
        # starts, ends = starts[vertices], ends[vertices]
        starts, ends = self.row_ptr[vertices], self.row_ptr[vertices + 1]
        sizes = (ends - starts)
        ranges = torch.arange(sizes.sum(), device=self.device)
        indices = ranges + starts.repeat_interleave(sizes) - (sub_row_ptr[:-1]).repeat_interleave(sizes)
        sub_columns = self.columns[indices]
        
        # fetch attributes
        sub_vertex_attrs_tensor, sub_vertex_attrs_mask = None, None
        # print('self.vertex_attrs_tensor: {}'.format(self.vertex_attrs_tensor))
        # print('vertices:{}'.format(vertices))
        if self.vertex_attrs_tensor is not None:
            sub_vertex_attrs_tensor = self.vertex_attrs_tensor.index_select(1, vertices)
            sub_vertex_attrs_mask = self.vertex_attrs_mask.index_select(1, vertices)
        # print('self.vertex_attrs_tensor: {}'.format(self.vertex_attrs_tensor))
        sub_edge_attrs_tensor, sub_edge_attrs_mask = None, None
        if self.edge_attrs_tensor is not None:
            sub_edge_attrs_tensor = self.edge_attrs_tensor.index_select(1, indices)
            sub_edge_attrs_mask = self.edge_attrs_mask.index_select(1, indices)
        
        return CSRGraph(sub_columns, sub_row_ptr, self.directed, self.vertex_attrs_list,
                        sub_vertex_attrs_tensor, sub_vertex_attrs_mask,
                        self.edge_attrs_list, sub_edge_attrs_tensor, sub_edge_attrs_mask), indices
        
    def subgraph(self, vertices: torch.Tensor):
        # map
        new_vertices_to_old = vertices.sort().unique_consecutive()
        old_vertices_to_new = {}
        for i, v in enumerate(new_vertices_to_old):
            old_vertices_to_new[v] = i
        # to LIL
        all_nbrs = []
        new_nbrs_list = []
        lengths = [0]
        for i in range(len(self.row_ptr) - 1):
            all_nbrs = self.columns[self.row_ptr[i]:self.row_ptr[i+1]]
        # leave specified vertices in LIL
        for nbrs in all_nbrs:
            nbrs = nbrs[torch.where(nbrs in vertices)[0]]
            for i, e in enumerate(nbrs):
                nbrs[i] = old_vertices_to_new[e]
            new_nbrs_list.append(nbrs)
            lengths.append(len(nbrs))
        # LIL to CSR
        new_nbrs = torch.cat(new_nbrs_list)
        ptr = torch.tensor(lengths, dtype=torch.int32, device=self.device).cumsum(0)
        return CSRGraph(new_nbrs, ptr), new_vertices_to_old

    def get_vertex_attr(self, vertices, attr):
        assert torch.all(vertices < self.num_vertices)
        attr_id = self.vertex_attrs_map[attr]
        return self.vertex_attrs[attr_id][vertices]
    
    def select_vertex_by_attr(self, attr, cond):
        attr_id = self.vertex_attrs_map[attr]
        return torch.where(cond(self.vertex_attrs[attr_id]))[0]
    
    def set_vertex_attr(self, vertices, attr, value, mask):
        assert torch.all(vertices < self.num_vertices)
        attr_id = self.vertex_attrs_map[attr]
        self.vertex_attrs[attr_id][vertices] = torch.where(mask, value, self.vertex_attrs[attr_id][vertices])
    
    def get_edge_attr(self, edges, attr):
        assert torch.all(edges < self.num_edges)
        attr_id = self.edge_attrs_map[attr]
        return self.edge_attrs[attr_id][edges]
    
    def select_edge_by_attr(self, attr, cond):
        attr_id = self.edge_attrs_map[attr]
        return torch.where(cond(self.edge_attrs[attr_id]))[0]
    
    def set_edge_attr(self, edges, attr, value, mask):
        assert torch.all(edges < self.num_edges)
        attr_id = self.edge_attrs_map[attr]
        self.edge_attrs[attr_id][edges] = torch.where(mask, value, self.edge_attrs[attr_id][edges])

    @staticmethod
    def edge_list_to_Graph(edge_starts, edge_ends, directed=False, vertices=None, edge_attrs=None, edge_attrs_list=[], vertex_attrs=None, vertex_attrs_list=[]):
        """
        Read edgelists and return an according CSRGraph.
        :param np.array edge_starts: starting points of edges
        :param np.array edge_ends: ending points of edges
        :param bool directed: whether the graph is directed
        :param np.array vertices: vertices. can be None
        :param List[np.array] edge_attrs: a list data for each edge
        :param List edge_attrs_list: a list of edge attributes (preferably strings, like names of the attributes)
        :param List[np.array] vertex_attrs: a list data for each vertex (in the same order as vertices. please don't set vertices=None if you use this)
        :param List vertex_attrs_list: a list of vertex attributes (preferably strings, like names of the attributes)
        :return: CSRGraph, a dictionary of vertex to index, and a list of edge data in Tensor and CSR order
        """
        #得先删除重边，再按照度排序

        if vertices is None:
            vertices = np.array([], dtype=np.int32)
            for s, d in zip(edge_starts, edge_ends):
                vertices = np.append(vertices, s)
                vertices = np.append(vertices, d)
            # vertices = np.unique(vertices)  #这是按照原始顶点边号由小到大排序的，如何增加顶点度排序
            vertices, counts = np.unique(vertices, return_counts=True)#np.sort返回一个新的数组，其中的元素按升序排列（从小到大）
            idx = np.argsort(counts)
            vertices = vertices[idx]
        # print("排序后的顶点顺序：", vertices)
        
        # get vertex to index mapping
        vertex_to_index = {}
        vertex_data_list = [[] for _ in range(len(vertex_attrs_list))]
        for vertex, index in zip(vertices, range(len(vertices))):
            vertex_to_index[vertex] = index
            if vertex_attrs is not None:
                for data_index, data in enumerate(vertex_attrs):
                    vertex_data_list[data_index].append(data[vertex])
        # sort edge lists into val, col_ind, and row_ind
        num_vertices = len(vertices)
        num_data = len(edge_attrs)
        col_ind_list = [[] for _ in range(num_vertices)]
        data_list = [[[] for _ in range(num_vertices)] for _ in range(num_data)]
        # print("edge_starts", edge_starts)
        # print("edge_ends", edge_ends)
        for start, end, *data in zip(edge_starts, edge_ends, *edge_attrs):
            start_v = vertex_to_index[start]
            end_v = vertex_to_index[end]
            if start_v > end_v:    #确保取一半图数据时，起点都是小于终点的
                temp = start_v
                start_v = end_v
                end_v = temp
            col_ind_list[start_v].append(end_v)
            if not directed:
                col_ind_list[end_v].append(start_v)
            for d in range(num_data):
                data_list[d][start_v].append(data[d])
                if not directed:
                    data_list[d][end_v].append(data[d])    

        # if not directed:  # unique  #也对权重数据去重
        for i in range(len(col_ind_list)):
            col_ind_list[i], indices = np.unique(col_ind_list[i], return_index=True) #除重复的元素，然后将剩余元素（邻居节点）按升序排列。
            # print("indices", indices)
            for d in range(num_data):
                # print("data_list[d][i][indices]", data_list[d][i][indices])
                data_list[d][i] = np.array(data_list[d][i])[indices].tolist()
            # data_list[0 :][i][1:] = data_list[0 :][i][indices]
            
        col_ind = torch.zeros(sum([len(l) for l in col_ind_list]), dtype=torch.int32)
        # print("col_ind type: ", col_ind.dtype)
        row_ind = torch.zeros(num_vertices + 1, dtype=torch.int32)
        data_tensor = [torch.zeros(sum([len(l) for l in col_ind_list]), dtype=torch.int32) for _ in range(num_data)]
        curr_index = 0
        for l, v, *d in zip(col_ind_list, range(num_vertices), *data_list):
            col_ind[curr_index:curr_index + len(l)] = torch.tensor(l, dtype=torch.int32)
            row_ind[v] = curr_index
            for d2 in range(num_data):
                data_tensor[d2][curr_index:curr_index + len(l)] = torch.tensor(d[d2], dtype=torch.float32)
            curr_index += len(l)
        row_ind[-1] = curr_index
        if len(data_tensor) != 0:
            edge_attrs_tensor = torch.stack(data_tensor, dim=0)
            edge_attrs_mask = torch.ones(edge_attrs_tensor.shape, dtype=torch.bool)
        else:
            edge_attrs_tensor, edge_attrs_mask = None, None
        if vertex_attrs is not None:
            vertex_attrs_tensor = torch.stack([torch.tensor(l, dtype=torch.float32) for l in vertex_data_list], dim=0)
            vertex_attrs_mask = torch.ones(vertex_attrs_tensor.shape, dtype=torch.bool)
        else:
            vertex_attrs_tensor = None
            vertex_attrs_mask = None
        # print('CSRGraph: col_ind: {}, row_ind: {}, directed: {}, vertex_attr_list: {}, vertex_attrs_tensor: {}'.format(col_ind, row_ind, directed, vertex_attrs_list, vertex_attrs_tensor))
        # print('edge_attrs_list:{}, edge_attrs_tensor:{}, edge_attrs_mask:{}'.format(edge_attrs_list, edge_attrs_tensor, edge_attrs_mask))
        return CSRGraph(col_ind, row_ind, directed, vertex_attrs_list, 
                        vertex_attrs_tensor, vertex_attrs_mask, edge_attrs_list,
                        edge_attrs_tensor, edge_attrs_mask), vertex_to_index
        
    @staticmethod
    def read_graph(f, split=None, directed=False, edge_attrs_list=[]):
        """
        Read an edgelist file and return an according CSRGraph.
        Edge lists should has the following format:
        v_0[split]v_1
        values will default to .0.
        By default, graphs are stored in CPU.
        
        :param str f: filename for edge list
        :param str split: split string for each line
        :param bool directed: whether the graph is directed
        :return: CSRGraph and a dictionary of vertex to index
        """
        edge_starts, edge_ends, vertices, data = Graph.read_edgelist(f, split, directed)
        return CSRGraph.edge_list_to_Graph(edge_starts, edge_ends, directed, vertices, edge_attrs=data, edge_attrs_list=edge_attrs_list)
