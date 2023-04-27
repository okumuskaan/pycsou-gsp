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

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.deps as pycd
import pycsou.runtime as pycrt

__all__ = [
    "GraphGradient",
    "GraphDivergence",
    "GraphLaplacian",
]

def _sparse_to_rcd(arr):
    spi = pycd.SparseArrayInfo.from_obj(arr)
    if (spi == pycd.SparseArrayInfo.SCIPY_SPARSE) or (spi == pycd.SparseArrayInfo.CUPY_SPARSE):
        sp = spi.module()
        if not sp.isspmatrix_coo(arr):
            arr = arr.tocoo()
        row = arr.row
        col = arr.col
        data = arr.data
    elif (spi == pycd.SparseArrayInfo.PYDATA_SPARSE):
        arr.asformat("coo")
        row, col = arr.coords
        data = arr.data
    else:
        raise ValueError("Unknown input for sparse array.")
    return (row, col, data)
    
    
    
class _GraphOpBase(object):
    """
    Base Class for converting adjecency matrix W to row, col and data arrays that are module-agnostic according to sample_arr and input array of apply and adjoint.
    
    Notes
    -----
    Inside the init constructor, call ``_convert_W_to_rcd`` and get rcd: W --> (wrow, wcol, wdata) data. W can be either array of NumPy, DaskArray, CuPy or sparse array of SciPy Sparse, PyData Sparse, Cupy Sparse.
    
    Then, inside the init constructor, call ``_init_update_rcd`` method to convert rcd arrays to modules according to input sample_arr.
    
    Call ``_apply_update_rcd`` method to convert rcd arrays to modules according to input array of apply or adjoint function. If the modules are equivalent to the stored rcd array's module, then nothing is done.
    
    
    """

    # Input is W, weighted adjacency matrix.
    
    def _convert_W_to_rcd(self, W):
        """
        Converts weighted adjacency matrix to row, col, data arrays.
        
        If the matrix is sparse array, then _sparse_to_rcd is called
        If the matrix is array, then nonzero method and accessing is applied
        As a result, (wrow, wcol, wdata) is returned.
        
        """
        if (type(W) in pycd.supported_array_types()):
            _wrow, _wcol = W.nonzero()
            _wdata = W[_wrow, _wcol]
        else:
            _wrow, _wcol, _wdata = _sparse_to_rcd(W)
            
        return (_wrow, _wcol, _wdata)
            
    
    def _init_update_rcd(self, sample_arr):
        """
        Converts the modules of arrays wrow, wcol, wdata same as that of module input sample array
        
        This should be called inside the init.
        
        """
        self._Ninfo = pycd.NDArrayInfo
        if sample_arr is not None:
            ndo = self._Ninfo.from_obj(sample_arr)
            self._update_rcd(ndo)
        else:
            self._backup = None # Data stored in NUMPY
            
    def _apply_update_rcd(self, x):
        """
        Checks if the module of input array of apply or adjoint function is equal to the stored arrays. If not, update row, col, data
        """
        ndo = self._Ninfo.from_obj(x)
        print(ndo)
        if (ndo != self._backup):
            print("IF YES")
            self._update_rcd(ndo)
        else:
            print("IF NO")

    def _update_rcd(self, ndo):
        """
        Convert the arrays (wrow, wcol, wdata) to the module of input array
        """
        if ndo == self._Ninfo.DASK:
            # Convert (wrow, wcol, wdata) to dask array
            # CUPY --> DASK and NUMPY --> DASK
            xp = ndo.module()
            if self._backup == self._Ninfo.CUPY:
                # Extra operation for CUPY --> DASK
                self._wrow = self._wrow.get()
                self._wcol = self._wcol.get()
                self._wdata = self._wdata.get()
            self._wrow = xp.from_array(self._wrow)
            self._wcol = xp.from_array(self._wcol)
            self._wdata = xp.from_array(self._wdata)
            self._backup = self._Ninfo.DASK
                
        elif ndo == self._Ninfo.CUPY:
            # Convert (wrow, wcol, wdata) to cupy array
            # NUMPY --> CUPY and DASK --> CUPY
            xp = ndo.module()
            if self._backup == self._Ninfo.DASK:
                # Extra operation for DASK --> CUPY
                self._wrow = self._wrow.compute()
                self._wcol = self._wcol.compute()
                self._wdata = self._wdata.compute()
            self._wrow = xp.array(self._wrow)
            self._wcol = xp.array(self._wcol)
            self._wdata = xp.array(self._wdata)
            self._backup = self._Ninfo.CUPY

        elif ndo == self._Ninfo.NUMPY:
            # Convert (wrow, wcol, wdata) to numpy array
            if self._backup == None:
                # None is NUMPY --> NUMPY
                pass
            else:
                # DASK --> NUMPY and CUPY --> NUMPY
                if self._backup == self._Ninfo.DASK:
                    # DASK --> NUMPY
                    self._wrow = self._wrow.compute()
                    self._wcol = self._wcol.compute()
                    self._wdata = self._wdata.compute()
                elif self._backup == self._Ninfo.CUPY:
                    # CUPY --> NUMPY
                    self._wrow = self._wrow.get()
                    self._wcol = self._wcol.get()
                    self._wdata = self._wdata.get()
                self._backup = self._Ninfo.NUMPY

        else:
            raise ValueError("Unknown type of sample array")
        
    
    

class GraphGradient(pyca.LinOp, _GraphOpBase):
    r"""
    Graph gradient operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph gradient, :math:`\nabla : \mathbb{R}^N \rightarrow \mathbb{R}^{N_e}` , is defined as
    
    .. math::
        {(\nabla \mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
        
    This is the approximation of the first derivative of a signal using finite-differences on irregular domain such as graphs.
        
    Adjoint of graph gradient, :math:`\nabla^* : \mathbb{R}^{N_e} \rightarrow \mathbb{R}^{N}`, is graph divergence:
    
    .. math::
        {(\nabla^*\mathbf{F})_i = \sum_{j \in \mathcal{V}} 2 \sqrt{w_{ij}} \mathbf{F}_{ij}}
    
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
    
    
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray],
                 sample_arr: typ.Optional[pyct.NDArray] = None):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        sample_arr: ``pyct.NDArray`` or ``None``
            Optional sample input array to convert data module. Default: ``None``.
        """
        
        self._wrow, self._wcol, self._wdata = self._convert_W_to_rcd(W)
        self._backup = None
        
        self._N = W.shape[0]
        self._Ne = len(self._wdata)
        
        self._init_update_rcd(sample_arr) # self._backup and self._wrow, self._wcol, self._data updated
        
        super().__init__(shape=(self._Ne , self._N))
        
        #self._lipschitz = ...
        
    
    
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array with :math:`N` elements.
        
        Returns
        -------
        ``pyct.NDArray``
            Output of gradient array with :math:`N_e` elements.
        """
        self._apply_update_rcd(arr)
        diff = arr[self._wrow] - arr[self._wcol]
        return diff * (self._wdata**0.5)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:#: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array with :math:`N_e` elements.
            
        Returns
        -------
        ``pyct.NDArray``
            Output of adjoint of gradient array with `N` elements.
        """
        self._apply_update_rcd(arr)
        xp = pycu.get_array_module(arr)
        y = xp.zeros((self._N,))
        return self._sum_diff_vertex(self._wrow, arr * 2 * (self._wdata**0.5), y) # it's Graph Divergence
    
        
    @staticmethod
    @nb.jit(parallel=True, nopython=True)
    def _sum_diff_vertex(row, diff, y):
        for i in range(row.shape[0]):
            y[row[i]] += diff[i]
        return y
    
    
    
class GraphDivergence(pyca.LinOp):
    r"""
    Graph divergence operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    Given a graph signal vector :math:`\mathbf{F} \in \mathbf{R}^{N_e}`, where :math:`N_e = |\mathcal{E}|`, the graph divergence, :math:`\text{div} : \mathbb{R}^{N_e} \rightarrow \mathbb{R}^N` , is defined as

    .. math::
        {(\text{div} \mathbf{F})_{i} = \sum_{j \in \mathcal{V}} 2\sqrt{w_{ij}} \mathbf{F}_{ij}}
        
    Adjoint of graph divergence, :math:`\text{div}^* : \mathbb{R}^N \rightarrow \mathbb{R}^{N_e}`, is graph gradient:
    
    .. math::
        {(\text{div}^* \mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
    
    See Also
    --------
    ``GraphGradient``
    """
    
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray],
                 sample_arr: typ.Optional[pyct.NDArray] = None):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        sample_arr: ``pyct.NDArray`` or ``None``
            Optional sample input array to convert data module. Default: ``None``.
        """
        self._GraphGrad = GraphGradient(W, sample_arr)
        super().__init__(shape=(self._GraphGrad._N, self._GraphGrad._Ne))
        #self._lipschitz = ...
        

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array with :math:`N_e` elements.
        
        Returns
        -------
        ``pyct.NDArray``
            Output of divergence array with :math:`N` elements.
        """
        return self._GraphGrad.adjoint(arr)
    
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray):
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array with :math:`N` elements.
            
        Returns
        -------
        ``pyct.NDArray``
            Output of adjoint of divergence array with `N_e` elements.
        """
        return self._GraphGrad(arr)


class GraphLaplacian(pyca.SelfAdjointOp, _GraphOpBase):
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
    
        #import pycgsp.linop.diff as pycgspd
        #import pycgsp.core.graph as pycgspg
        #import pycgsp.core.plot as pycgspp
        #import pygsp.graphs as pygspg
        #import matplotlib.pyplot as plt
        #import numpy as np
        
        #G2 = pygspg.Minnesota()
        #G1 = pycgspg.Graph(G2.W)
        #G1Lap = pycgspd.GraphLaplacian(G1)
        #G2Lap = pycgspd.GraphLaplacian(G2)
        #vec = np.random.randn(G2.N,)
        #lap_arr_1 = G1Lap(vec)
        #lap_arr_2 = G2Lap(vec)
        #G2.compute_laplacian()
        #lap_arr_pygsp = G2.L.dot(vec)
        #np.allclose(lap_arr_1, lap_arr_2)
        #np.allclose(lap_arr_1, lap_arr_pygsp)
        #fig,ax = plt.subplots(1, 2, figsize=(10,4))
        #pycgspp.myGraphPlotSignal(G2, s=vec, title="Input Signal", ax=ax[0])
        #pycgspp.myGraphPlotSignal(G2, s=lap_arr_1, title="Laplacian of Signal by Pycgsp", ax=ax[1])
        #plt.show()
    
    """
    
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray],
                 lap_type="combinatorial",
                 sample_arr: typ.Optional[pyct.NDArray] = None):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        lap_type: ``str``
            Laplacian type. Default: combinatorial.
        sample_arr: ``pyct.NDArray`` or ``None``
            Optional sample input array to convert data module. Default: ``None``.
            
        TODO: Degree Matrix Needed for Normalized Laplacian or It can be computed here
        """
        
        self._wrow, self._wcol, self._wdata = self._convert_W_to_rcd(W)
        
        self._N = W.shape[0]
        
        self._lap_type = lap_type
        
        self._init_update_rcd(sample_arr) # self._backup and self._wrow, self._wcol, self._data updated
        
        super().__init__(shape=self.W.shape)
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array with ``N`` elements.
            
        Returns
        -------
        ``pyct.NDArray``
            Output array with ``N`` elements.
        """
        self._apply_update_rcd(arr)
        xp = pycu.get_array_module(arr)
        y = xp.zeros((self._N,))
        
        if self._lap_type == "combinatorial":
            diff = arr[self._wrow] - arr[self._wcol]
            weighted_diff = diff * self._wdata
            return self._sum_diff_vertex(self._wrow, weighted_diff, y)
        else:
            raise NotImplementedError("Matrix-free Normalized Laplacian Implementation not supported yet")
    
    @staticmethod
    @nb.jit(parallel=True, nopython=True)
    def _sum_diff_vertex(row, diff, y):
        for i in range(len(row)):
            y[row[i]] += diff[i]
        return y
    

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
