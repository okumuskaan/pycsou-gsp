# #############################################################################
# diff.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Graph differential operators.

This module provides various graph differential operators that are based on ``pycsou.abc.core.LinOp``.

.. rubric:: Classes for Differential Operators

.. autosummary::

  GraphGradient
  GraphDivergence
  GraphLaplacian
  GraphHessian

"""

import typing as typ

import numba as nb
import numpy as np

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.runtime as pycrt
import pycgsp.core as pycgspc
import pygsp


class GraphGradient(pyca.LinOp):
    r"""
    Graph gradient operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    Given a graph signal :math:`\mathbf{f} \in \mathbf{R}^N`, where :math:`N = |\mathcal{V}|`, the graph gradient, :math:`\nabla_{\mathcal{G}} : \mathbb{R}^N \rightarrow \mathbb{R}^{N \times N}` , is defined as
    
    .. math::
        {(\nabla_{\mathcal{G}}\mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
        
    This is the approximation of the first derivative of a signal using finite-differences on irregular domain such as graphs.
        
    Adjoint of graph gradient, :math:`\nabla^*_{\mathcal{G}} : \mathbb{R}^{N \times N} \rightarrow \mathbb{R}^{N}`, is graph divergence:
    
    .. math::
        {(\nabla^*_{\mathcal{G}}\mathbf{F})_i = \sum_{j \in \mathcal{V}} \sqrt{w_{ij}} \mathbf{F}_{ij}}
    
    Examples
    --------
    >>> import pycgsp.linop.diff as pycgspd
    >>> import pycgsp.core.graph as pycgspg
    >>> import pycgsp.core.plot as pycgspp
    >>> import pygsp.graphs as pygspg
    >>> import numpy as np
    >>> vec = np.arange(5)
    >>> W = np.array([[0,2,0,3,0], [2,0,0,0,0], [0,0,0,1,0], [3,0,1,0,2], [0,0,0,2,0]])
    >>> G1 = pycgspg.Graph(W)
    >>> G2 = pygspg.Graph(W)
    >>> G1Grad = pycgspd.GraphGradient(G1)
    >>> G2Grad = pycgspd.GraphGradient(G2)
    >>> grad_arr_1 = G1Grad(vec)
    >>> grad_arr_2 = G2Grad(vec)
    >>> G2.compute_differential_operator()
    >>> grad_arr_pygsp = G2.grad(vec)
    >>> np.allclose(grad_arr_1, grad_arr_2)
    True
    >>> data_pos = grad_arr_1[grad_arr_1>0]
    >>> np.allclose(data_pos, grad_arr_pygsp)
    True
    
    See Also
    --------
    ``GraphDivergence``
    """
    #GraphSpec = typ.Union[pycgspc.Graph, pygsp.graphs.Graph]
    
    
    def __init__(self, Graph):#: typ.Union[pycgspc.graph.Graph, pygsp.graphs.Graph]):#: GraphSpec):
        r"""
        Parameters
        ----------
        Graph: ``pycgsp.core.Graph`` or ``pygsp.graphs.Graph``
            Graph object.
        """
        
        self.W = Graph.W.tocoo()
        self._out = self.W.copy()
        self._adj_out = None
        self._N = Graph.N
        
        super().__init__(shape=(Graph.Ne, Graph.N))
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array.
        
        Returns
        -------
        Sparse.coo_matrix
            Output of divergence array
        """
        xp = pycu.get_array_module(arr)
        self._adj_out = xp.zeros((self._N,))
        diff = arr[self.W.row] - arr[self.W.col]
        self._out.data = diff * (self.W.data**0.5)
        return self._out

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr) -> pyct.NDArray:#: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: Sparse array
            Input array.
            
        Returns
        -------
        ``pyct.NDArray``
            Output of adjoint of divergence array
        """
        arr = arr.tocoo()
        if self._adj_out is None:
            self._adj_out = np.zeros((self._N,))
        else:
            self._adj_out *= 0
        return self._sum_diff_vertex(self.W.row, arr.data * (self.W.data**0.5), self._adj_out) # it's Graph Divergence
    
        
    @nb.jit(parallel=True, forceobj=True)
    def _sum_diff_vertex(self, row, diff, y):
        for i in range(len(row)):
            y[row[i]] += diff[i]
        return y
    
    
    
class GraphDivergence(pyca.LinOp):
    r"""
    Graph divergence operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    Given a graph signal vector :math:`\mathbf{F} \in \mathbf{R}^{N \times N}`, where :math:`N = |\mathcal{V}|`, the graph divergence, :math:`\text{div}_{\mathcal{G}} : \mathbb{R}^{N \times N} \rightarrow \mathbb{R}^N` , is defined as

    .. math::
        {(\text{div}_{\mathcal{G}}\mathbf{F})_{i} = \sum_{j \in \mathcal{V}} \sqrt{w_{ij}} \mathbf{F}_{ij}}
        
    Adjoint of graph gradient, :math:`\text{div}^*_{\mathcal{G}} : \mathbb{R}^N \rightarrow \mathbb{R}^{N \times N}`, is graph gradient:
    
    .. math::
        {(\text{div}^*_{\mathcal{G}}\mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
    
    See Also
    --------
    ``GraphGradient``
    """
    
    def __init__(self, Graph):
        r"""
        Parameters
        ----------
        Graph: ``pycgsp.core.Graph`` or ``pygsp.graphs.Graph``
            Graph object.
        """
        self._GraphGrad = GraphGradient(Graph)
        super().__init__(shape=(Graph.N, Graph.Ne))
        

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr) -> pyct.NDArray:#: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: Sparse array
            Input array.
            
        Returns
        -------
        ``pyct.NDArray``
            Output of divergence array
        """
        return self._GraphGrad.adjoint(arr)
    
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array.
        
        Returns
        -------
        Sparse.coo_matrix
            Output of adjoint of divergence array
        """
        return self._GraphGrad(arr)


class GraphLaplacian(pyca.SelfAdjointOp):
    r"""
    Graph laplacian operator.
    
    Bases: ``pycsou.abc.operator.SelfAdjointOp``
            
    Given a graph signal :math:`\mathbf{f} \in \mathbf{R}^N`, where :math:`N = |\mathcal{V}|`, the graph laplacian, :math:`L : \mathbb{R}^N \rightarrow \mathbb{R}^N` , is defined as
    
    .. math::
        {(L \mathbf{f})_{i} = \sum_{j \in \mathcal{V}} w_{ij} (\mathbf{f}_i - \mathbf{f}_j)}

    Notes
    -----
    Graph Laplacian is self adjoint operator.
    
    Examples
    --------
    
    .. plot::
    
        import pycgsp.linop.diff as pycgspd
        import pycgsp.core.graph as pycgspg
        import pycgsp.core.plot as pycgspp
        import pygsp.graphs as pygspg
        import matplotlib.pyplot as plt
        import numpy as np
        
        G2 = pygspg.Minnesota()
        G1 = pycgspg.Graph(G2.W)
        G1Lap = pycgspd.GraphLaplacian(G1)
        G2Lap = pycgspd.GraphLaplacian(G2)
        vec = np.random.randn(G2.N,)
        lap_arr_1 = G1Lap(vec)
        lap_arr_2 = G2Lap(vec)
        G2.compute_laplacian()
        lap_arr_pygsp = G2.L.dot(vec)
        np.allclose(lap_arr_1, lap_arr_2)
        np.allclose(lap_arr_1, lap_arr_pygsp)
        fig,ax = plt.subplots(1, 2, figsize=(10,4))
        pycgspp.myGraphPlotSignal(G2, s=vec, title="Input Signal", ax=ax[0])
        pycgspp.myGraphPlotSignal(G2, s=lap_arr_1, title="Laplacian of Signal by Pycgsp", ax=ax[1])
        plt.show()
    
    """
    
    def __init__(self, Graph, lap_type="combinatorial"):
        r"""
        Parameters
        ----------
        Graph: ``pycgsp.core.Graph`` or ``pygsp.graphs.Graph``
            Graph object.
        lap_type: ``str``
            Laplacian type. Default: combinatorial.
        """
        
        if lap_type == Graph.lap_type:
            pass
            #print("[INFO]: lap_type consistent")
        else:
            Graph.compute_laplacian(lap_type=lap_type)
        
        self.W = Graph.W
        self.L = Graph.L.tocoo()
        self._lap_type = lap_type
        
        super().__init__(shape=self.W.shape)
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array.
            
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        if self._lap_type == "combinatorial":
            xp = pycu.get_array_module(arr)
            diff = arr[self.L.row] - arr[self.L.col]
            weighted_diff = diff * self.W[self.L.row, self.L.col].toarray()[0]
            return self._sum_diff_vertex(self.L.row, weighted_diff, xp.zeros_like(arr))
        else:
            raise NotImplementedError("Matrix-free Normalized Laplacian Implementation not supported yet")
    
    @nb.jit(parallel=True, forceobj=True)
    def _sum_diff_vertex(self, row, diff, y):
        for i in range(len(row)):
            y[row[i]] += diff[i]
        return y

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array.
            
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        return self(arr) # since it's self-adjoint
    

class GraphHessian(pyca.LinOp):
    r"""
    Graph hessian operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    TODO: Solve this.
    """
    
    def __init__(self, Graph):
        
        self.W_lil = Graph.W
        self.W = Graph.W.tocoo()
        self._N = Graph.N
        
        super().__init__(shape=(Graph.Ne*Graph.Ne, Graph.N))
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        xp = pycu.get_array_module(arr)
        # List of sparse matrices
        H = self._sum_diff_vertex(arr, self._N, self.W.row, self.W.col, [], xp)
        return self._out

    @nb.jit(parallel=True, forceobj=True)
    def _sum_diff_vertex(self, arr, N, row, col, H, xp):
        for i in range(N):
            sub_col = col[row==i]
            sub_col_x, sub_col_y = xp.meshgrid(sub_col, sub_col)
            sub_col_x = xp.ravel(sub_col_x)
            sub_col_y = xp.ravel(sub_col_y)
            H.append(((arr[i] - arr[sub_col_x]) * self.W_lil()[i,sub_col_x] + (arr[row[i]] - arr[sub_col_y]) * self.W_lil()[i,sub_col_y])/2)
        return H

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):#: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        raise NotImplementedError("Adjoint of Graph Hessian Not Supported!")
