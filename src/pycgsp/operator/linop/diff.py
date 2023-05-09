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

    

class GraphGradient(pyca.LinOp):
    r"""
    Graph gradient operator.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph gradient, :math:`\nabla : \mathbb{R}^N \rightarrow \mathbb{R}^{N_e}` , is defined as
    
    .. math::
        {(\nabla \mathbf{f})_{ij} = \sqrt{w_{ij}} (\mathbf{f}_i - \mathbf{f}_j)}
        
    This is the approximation of the first derivative of a signal using finite-differences on irregular domain such as graphs.
        
    Adjoint of graph gradient, :math:`\nabla^* : \mathbb{R}^{N_e} \rightarrow \mathbb{R}^{N}`, is graph divergence:
    
    .. math::
        {(\nabla^*\mathbf{F})_i = \sum_{j \in \mathcal{V}} 2 \sqrt{w_{ij}} \mathbf{F}_{ij}}
        
    **Implementation Notes**
    
    The instances are not arraymodule-agnostic: they will only work with NDArrays belonging to the same array module as attributes of sparse arrays of ``W``.
        If W is cupy.sparse, the backend arraymodule is CUPY.
        If W is either sparse or scipy.sparse, the backend arraymodule is NUMPY
    
    In apply and adjoint operator, the module of the input array is expected to be equal to backend arraymodule. If not, warning is given, and non-efficient conversion is applied to make it module agnostic.
    
    Examples
    --------
    >>> import pycgsp.operator.linop.diff as pycgspd
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
    
    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True
    ):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weight matrix of a graph.
        enable_warnings: ``bool``
            Boolean data that determines if warnings about the compatibility of input array with backend are created or not. Default: ``True``.
        """
        
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        
        self._N = W.shape[0]
        self._Ne = len(self._wdata)
        super().__init__(shape=(self._Ne , self._N))
                
        self._dtype = self._wdata[0].dtype
        self._arraymodule = pycd.NDArrayInfo.from_obj(self._wdata)
        self._enable_warnings = bool(enable_warnings)
        
        self._compute_div = self._create_div_func(self._arraymodule)
        #self._lipschitz = ...
        
    @staticmethod
    def _create_div_func(ndi):
        if (ndi == pycd.NDArrayInfo.NUMPY):
            _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def div_jit(rows, ws, arr, res):
    for i, row in enumerate(rows):
        res[row] += (ws[i]**0.5) * arr[i]
    return res
"""
        elif (ndi == pycd.NDArrayInfo.CUPY):
            _code = r"""
import numba.cuda as nbcuda

@nbcuda.jit(device=True, fastmath=True, opt=True)
def div_jit(rows, ws, arr, res):
    for i, row in enumerate(rows):
        res[row] += (ws[i]**0.5) * arr[i]
    return res
"""
        exec(_code, locals())
        return eval("div_jit")
    
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
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        res = (arr[self._wrow] - arr[self._wcol]) * (self._wdata**0.5)
        return pycgspu.convert_arr(res, self._arraymodule, ndi)

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
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        res = self._compute_div(self._wrow, self._wdata, arr, arr*0.0)
        return pycgspu.convert_arr(res, self._arraymodule, ndi)


            
        
    












    
    
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
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True
    ):
        r"""
        Parameters
        ----------
        W: ``pyct.NDArray`` or ``pyct.SparseArray``
            Weighted adjacency matrix of a graph.
        sample_arr: ``pyct.NDArray`` or ``None``
            Optional sample input array to convert data module. Default: ``None``.
        """
        self._GraphGrad = GraphGradient(W, enable_warnings)
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
    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        lap_type="combinatorial",
        enable_warnings: bool = True
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
        
        self._dtype = self._wdata[0].dtype
        self._arraymodule = pycd.NDArrayInfo.from_obj(self._wdata)
        self._enable_warnings = bool(enable_warnings)
        
        self._lapd = self._create_lapd(self._arraymodule, self._wrow, self._wdata, lap_type)
        #self._lap_type = lap_type
        
        self._compute_lap = self._create_lap_func(self._arraymodule)
        # self._lipschitz = ...
    
    @staticmethod
    def _create_lapd(ndi, wrow, wdata, lap_type):
        if lap_type == "combinatorial":
            xp = ndi.module()
            res = xp.ones(len(wdata), dtype=wdata[0].dtype)
        elif lap_type == "normalized":
            if ndi == pycd.NDArrayInfo.NUMPY:
                _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def deg_jit(rows, ws, res):
    for i, row in enumerate(rows):
        res[row] += ws[i]
    return res**(-0.5)
"""
            elif ndi == pycd.NDArrayInfo.CUPY:
                _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def deg_jit(rows, ws, res):
    for i, row in enumerate(rows):
        res[row] += ws[i]
    return res**(-0.5)
"""
            exec(_code, locals())
            lap = eval("lap_jit")
            res = lap(wrow, wdata, wdata*0.0)
        else:
            raise ValueError("Unknown lap_type. Either combinatorial or normalized must be given.")
        return res
    
    @staticmethod
    def _create_lap_func(ndi):
        if ndi == pycd.NDArrayInfo.NUMPY:
            _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def lap_jit(rows, cols, ws, arr, res, d):
    for i, row in enumerate(rows):
        res[row] += d[row] * ws[i] * (d[row]*arr[row] - d[cols[i]]*arr[cols[i]])
    return res
"""
        elif ndi == pycd.NDArrayInfo.CUPY:
            _code = r"""
import numba.cuda as nbcuda

@nbcuda.jit(device=True, fastmath=True, opt=True)
def lap_jit(rows, cols, ws, arr, res, d):
    for i, row in enumerate(rows):
        res[row] += d[row] * ws[i] * (d[row]*arr[row] - d[cols[i]]*arr[cols[i]])
    return res
"""
        exec(_code, locals())
        return eval("lap_jit")
        
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
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        res = self._compute_lap(self._wrow, self._wcol, self._wdata, arr, arr*0.0, self._lapd)
        return pycgspu.convert_arr(res, self._arraymodule, ndi)
 
 
    

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

