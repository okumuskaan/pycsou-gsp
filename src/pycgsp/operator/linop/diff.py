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
import warnings

import numpy as np
import numba as nb

import pycgsp.util as pycgspu

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

"""
@nb.jit(nopython=True, parallel=False, nogil=True, fastmath=True, cache=False)
def _compute_grad_NUMPY(arr, w, rows, cols):
    res = arr[rows]
    for i in range(len(rows)):
        res[i] = (res[i] - arr[cols[i]]) * w[i]
    return res

@nb.jit(nopython=True, parallel=False, nogil=True, fastmath=True, cache=False)
def _compute_div_NUMPY(arr, w, rows, cols, res):
    for i in range(len(rows)):
        res[rows[i]] += w[i] * arr[i]
        res[cols[i]] -= w[i] * arr[i]
    return res

@nb.guvectorize(
[(nb.float64[:], nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:]),
 (nb.float32[:], nb.float32[:], nb.int32[:], nb.int32[:], nb.float32[:])],
'(n), (m), (m), (m) -> (m)'
)
def _compute_grad_DASK(arr, w, rows, cols, res):
    res *= 0.0
    res = arr[rows]
    for i in range(len(rows)):
        res[i] = (res[i] - arr[cols[i]]) * w[i]

@nb.guvectorize(
[(nb.float64[:], nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:]),
 (nb.float32[:], nb.float32[:], nb.int32[:], nb.int32[:], nb.float32[:])],
'(n), (m), (m), (m) -> (m)'
)
def _compute_div_DASK(arr, w, rows, cols, res):
    for i in range(len(rows)):
        res[rows[i]] += w[i] * arr[i]
        res[cols[i]] -= w[i] * arr[i]

@nb.cuda.jit(device=True, fastmath=True, opt=True)
def _compute_grad_CUPY(arr, w, rows, cols):
    res = arr[rows]
    for i in nb.prange(len(rows)):
        res[i] -= arr[cols[i]]
        res[i] *= w[i]**0.5
    return res

@nb.cuda.jit(device=True, fastmath=True, opt=True)
def _compute_div_CUPY(arr, w, rows, cols, res):
    for i in nb.prange(len(rows)):
        res[rows[i]] += w[i] * arr[i]
        res[cols[i]] -= w[i] * arr[i]
    return res
    
@pycu.redirect(i="arr", NUMPY=_compute_grad_NUMPY, DASK=_compute_grad_DASK, CUPY=_compute_grad_CUPY)
def _compute_grad(arr, w, rows, cols):
    pass
    
@pycu.redirect(i="arr", NUMPY=_compute_div_NUMPY, DASK=_compute_div_DASK, CUPY=_compute_div_CUPY)
def _compute_div(arr, w, rows, cols):
    pass
"""



@nb.jit(nopython=True, parallel=False, nogil=True, fastmath=True, cache=False)
def compute_grad(arr, w, rows, cols):
    res = arr[rows]
    for i in range(len(rows)):
        res[i] = (res[i] - arr[cols[i]]) * w[i]
    return res

@nb.jit(nopython=True, parallel=False, nogil=True, fastmath=True, cache=False)
def compute_div(arr, w, rows, cols, res):
    for i in range(len(rows)):
        res[rows[i]] += w[i] * arr[i]
        res[cols[i]] -= w[i] * arr[i]
    return res




class GraphGradient(pyca.LinOp):
    r"""
    Graph gradient operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph gradient, :math:`\nabla : \mathbb{R}^N \rightarrow \mathbb{R}^{N_e}` , is defined as
    
    .. math::
        {(\nabla \mathbf{f})_{k: (i,j)} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
        
    This is the approximation of the first derivative of a signal using finite-differences on irregular domain such as graphs.
        
    Adjoint of graph gradient, :math:`\nabla^* : \mathbb{R}^{N_e} \rightarrow \mathbb{R}^{N}`, is graph divergence:
    
    .. math::
        {(\nabla^*\mathbf{F})_i = \sum_{j \in \mathcal{V}} 2 \sqrt{w_{k:(i,j)}} \mathbf{F}_{k:(i,j)}}
        
    **Implementation Notes**
    
    The instances are not arraymodule-agnostic: they will only work with NDArrays belonging to the same array module as attributes of sparse arrays of ``W``.
    If ``W`` is cupy.sparse, the backend arraymodule is ``CUPY``.
    If ``W`` is either sparse or scipy.sparse, the backend arraymodule is ``NUMPY``.
    
    In apply and adjoint operator, the module of the input array is expected to be equal to the backend arraymodule. If not, a warning is given, and a non-efficient conversion is applied to make it module agnostic.
    
    Examples
    --------
    
    
    See Also
    --------
    ``GraphDivergence``
    """

    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray]
    ):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        """
        
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        self._wdata = self._wdata**0.5

        self._N = W.shape[0]
        self._Ne = len(self._wdata)

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
        return compute_grad(arr, self._wdata, self._wrow, self._wcol)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
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
        xp = pycu.get_array_module(arr)
        res = xp.zeros(self._N, dtype=self._wdata.dtype)
        return compute_div(arr, self._wdata, self._wrow, self._wcol, res)





class GraphDivergence(pyca.LinOp):
    r"""
    Graph divergence operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    **Mathematical Notes**
    
    Given a graph signal vector :math:`\mathbf{F} \in \mathbf{R}^{N_e}`, where :math:`N_e = |\mathcal{E}|`, the graph divergence, :math:`\text{div} : \mathbb{R}^{N_e} \rightarrow \mathbb{R}^N` , is defined as

    .. math::
        {(\text{div} \mathbf{F})_{i} = \sum_{j \in \mathcal{V}} \sqrt{w_{ij}} \mathbf{F}_{ij}}
        
    Adjoint of graph divergence, :math:`\text{div}^* : \mathbb{R}^N \rightarrow \mathbb{R}^{N_e}`, is graph gradient:
    
    .. math::
        {(\text{div}^* \mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
    
    **Implementation Notes**
    
    The instances are not arraymodule-agnostic: they will only work with NDArrays belonging to the same array module as attributes of sparse arrays of ``W``.
    If W is cupy.sparse, the backend arraymodule is CUPY.
    If W is either sparse or scipy.sparse, the backend arraymodule is NUMPY
    
    In apply and adjoint operator, the module of the input array is expected to be equal to backend arraymodule. If not, warning is given, and non-efficient conversion is applied to make it module agnostic.
    
    See Also
    --------
    ``GraphGradient``
    """
    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray]
    ):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        sample_arr: ``pyct.NDArray`` or ``None``
            Optional sample input array to convert data module. Default: ``None``.
        """
        self._GraphGrad = GraphGradient(W)
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



@nb.jit(nopython=True, fastmath=True, nogil=True, cache=False)
def compute_lap_comb(arr, wrow, wcol, wdata):
    res = arr*0.0
    for i in range(len(wrow)):
        d = wdata[i] * (arr[wrow[i]] - arr[wcol[i]])
        res[wrow[i]] +=  d
        res[wcol[i]] -=  d
    return res

@nb.jit(nopython=True, fastmath=True, nogil=True, cache=False)
def compute_lap_norm(arr, wrow, wcol, wdata, a_sqrt):
    res = arr*0.0
    for i in range(len(wrow)):
        d = wdata[i] * (arr[wrow[i]]/a_sqrt[wrow[i]] - arr[wcol[i]]/a_sqrt[wcol[i]]) 
        res[wrow[i]] +=  d / a_sqrt[wrow[i]]
        res[wcol[i]] -= d / a_sqrt[wcol[i]]
    return res
    
class GraphLaplacian(pyca.SelfAdjointOp):
    r"""
    Graph laplacian operator.
    
    Bases: ``pycsou.abc.operator.SelfAdjointOp``
            
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbf{R}^N`, where :math:`N = |\mathcal{V}|`, the combinatorial graph laplacian, :math:`L : \mathbb{R}^N \rightarrow \mathbb{R}^N` , is defined as
    
    .. math::
        {(L \mathbf{f})_{i} = \sum_{j \in \mathcal{V}} w_{ij} (\mathbf{f}_i - \mathbf{f}_j)}
        
    Normalized graph laplacian, :math:`\tilde{L}` is defined as
    
    .. math::
        {(\tilde{L} \mathbf{f})_{i} = \frac{1}{\sqrt{d_i}} \sum_{j \in \mathcal{V}} w_{ij} (\frac{\mathbf{f}_i}{\sqrt{d_i}} - \frac{\mathbf{f}_j}{\sqrt{d_j}})}
    
    where :math:`d_i = \sum_{j \in \mathcal{V}} w_{ij}` is the degree of vertex :math:`i`.

    **Implementation Notes**
    
    The instances are not arraymodule-agnostic: they will only work with NDArrays belonging to the same array module as attributes of sparse arrays of ``W``.
        If W is cupy.sparse, the backend arraymodule is CUPY.
        If W is either sparse or scipy.sparse, the backend arraymodule is NUMPY
    
    In apply and adjoint operator, the module of the input array is expected to be equal to backend arraymodule. If not, warning is given, and non-efficient conversion is applied to make it module agnostic.

    Notes
    -----
    Graph Laplacian is self adjoint operator.
    
    Examples
    --------
    
    """
    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        lap_type="combinatorial"
    ):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        lap_type: ``str``
            Laplacian type. Default: combinatorial.
        enable_warnings: ``bool``
            Boolean data that determines if warnings about the compatibility of input array with backend are created or not. Default: ``True``.
        """
        
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        
        self._N = W.shape[0]
        super().__init__(shape=(self._N , self._N))
        
        self._lap_type = lap_type
        if lap_type=="normalized":
            self._a_sqrt = W.sum(0).A[0] ** 0.5
        #self._lipschitz = ...

    #@pycrt.enforce_precision(i="arr")
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
        if self._lap_type=="combinatorial":
            return compute_lap_comb(arr, self._wrow, self._wcol, self._wdata) 
        elif self._lap_type=="normalized":
            return compute_lap_norm(arr, self._wrow, self._wcol, self._wdata, self._a_sqrt)
        else:
            raise ValueError("Only combintorial or normalized laplacian type are supported.")
 
    

class GraphHessian(pyca.LinOp):
    r"""
    Graph hessian operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    TODO: Solve this.
    """
    
    def __init__(self, W):
        
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)

        self._N = len(self._wdata)
        super().__init__(shape=(self._N*self._N*self._N, Graph.N))
        
        
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        raise NotImplementedError("Graph Hessian Not Supported!")


    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):#: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        raise NotImplementedError("Adjoint of Graph Hessian Not Supported!")

