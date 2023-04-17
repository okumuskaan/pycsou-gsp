r"""
Abstract Graph Class for Pycsou-GSP

It's derived from the ``pygsp.graphs.Graph``.

"""

from pycgsp.linop.diff import *
import pycsou.runtime as pycrt
from scipy import sparse
import numpy as np

import pygsp

class Graph:
    r"""
    Graph Class.
    """
    
    def __init__(self, adjacency, lap_type="combinatorial", coords=None, plotting={}):
        # TODO: Not only with Numpy
        # Check adjacency data type
        if not sparse.isspmatrix(adjacency):
            adjacency = np.asanyarray(adjacency)
            
        self._adjacency = sparse.csr_matrix(adjacency, copy=False)
        # Checks and Errors for adjacency matrix, directed, numbers etc.
        
        self.n_vertices = self._adjacency.shape[0]
        
        self._adjacency.eliminate_zeros()
        
        self._directed = None
        self._connected = None
        
        #if self.is_directed():
        self.n_edges = self._adjacency.nnz #Assume it's directed now
        
        if coords is not None:
            self.coords = np.asanyarray(coords)
        
        
        self.plotting = {
            'vertex_size': 100,
            'vertex_color': (0.12, 0.47, 0.71, 0.5),
            'edge_color': (0.5, 0.5, 0.5, 0.5),
            'edge_width': 2,
            'edge_style': '-',
            'highlight_color': 'C1',
            'normalize_intercept': .25,
        }
        self.plotting.update(plotting)
        self.signals = dict()
        
        self._A = None
        self._d = None
        self._dw = None
        self._lmax = None
        self._lmax_method = None
        self._U = None
        self._e = None
        self._coherence = None
        self._D = None
        self._LapOp = None      # Pycsou LinOp - SelfAdjointOp
        self._GradOp = None
        self._DivOp = None
                
        self.lap_type = lap_type
        self.compute_laplacian(lap_type)
        # TODO: Write LapOp
        
        self.Ne = self.n_edges
        self.N = self.n_vertices
        
    @property
    def W(self):
        return self._adjacency.tolil()
    
    @W.setter
    def W(self, value):
        "TODO: Is it possible to update W ?"
        raise AttributeError('In-place modification of the graph is not '
                             'supported. Create another Graph object.')
    
    @property
    def A(self):
        if self._A is None:
            self._A = self.W > 0
        return self._A
    
    @property
    def d(self):
        if self._d is None:
            # Suppose undirected graph
            self._d = self.W.getnnz(axis=1)
        return self._d
    
    @property
    def dw(self):
        if self._dw is None:
            # Suppose undirected graph
            self._dw = np.ravel(self.W.sum(axis=0))
        return self._dw
        
    @property
    def LapOp(self):
        if self._LapOp is None:
            print("Creating GraphLaplacian Object...")
            self._LapOp = GraphLaplacian(Graph=self, lap_type=self.lap_type)
        return self._LapOp
        
    @property
    def GradOp(self):
        if self._GradOp is None:
            print("Creating GraphGradient Object...")
            self._GradOp = GraphGradient(Graph=self)
        return self._GradOp
        
    @property
    def DivOp(self):
        if self._DivOp is None:
            print("Creating GraphDivergence Object...")
            self._DivOp = GraphDivergence(Graph=self)
        return self._DivOp
        
    
    def compute_laplacian(self, lap_type='combinatorial'):
        if lap_type != self.lap_type:
            self._lmax = None
            self._U = None
            self._e = None
            self._coherence = None
            self._D = None
            self._LapOp = None
        
        self.lap_type = lap_type
        
        #Assume it's undirected:
        W = self.W
        
        if lap_type == 'combinatorial':
            D = sparse.diags(self.dw)
            self.L = D - W
        elif lap_type == 'normalized':
            d = np.zeros(self.n_vertices)
            disconnected = (self.dw == 0)
            np.power(self.dw, -0.5, where=~disconnected, out=d)
            D = sparse.diags(d)
            self.L = sparse.identity(self.n_vertices) - D.dot(W.dot(D))
            self.L[disconnected, disconnected] = 0
            self.L.eliminate_zeros()
        else:
            raise ValueError("Unknown Laplacian type {}".format(lap_type))

        
    @pycrt.enforce_precision(i="arr")
    def laplacian_op(self, arr, lap_type="combinatorial"):
        if lap_type != self.lap_type:
            self._lmax = None
            self._U = None
            self._e = None
            self._coherence = None
            self._D = None
            self._LapOp = None
        
        self.lap_type = lap_type
        
        return self.LapOp(arr)
        
    
    @pycrt.enforce_precision(i="arr")
    def grad_op(self, arr):
        return self.GradOp(arr)
            
    @pycrt.enforce_precision(i="arr")
    def div_op(self, arr):
        return self.DivOp(arr)


