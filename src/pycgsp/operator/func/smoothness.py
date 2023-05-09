r"""
Graph smoothness functionals.

.. rubric:: Classes for Smoothness Functionals

.. autosummary::

    LocalVariation
"""

import typing as typ

import pycsou.abc as pyca
import pycsou.util.ptype as pyct
import pycsou.runtime as pycrt

import pycgsp.util as pycgspu


class LocalVariation(pyca.SquareOp):
    r"""
    Graph local variation.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph local variation : :math:`\mathbb{R}^N \rightarrow \mathbb{R}^N` , is defined as
    
    .. math::
        {||\nabla_i f||_2 = (\sum_{j \in \mathcal{V}} w_{ij} (f_i - f_j)^2)^{1/2}}
    """
    
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True
    ):
        r"""
        """
        
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        self._N = W.shape[0]
        super().__init__(shape=(self._N , self._N))
        
        self._dtype = self._wdata[0].dtype
        self._arraymodule = pycd.NDArrayInfo.from_obj(self._wdata)
        self._enable_warnings = bool(enable_warnings)

        self._compute_locvar = self._create_locvar_func(self._arraymodule)
        
    @staticmethod
    def _create_locvar_func(ndi):
        if (ndi == pycd.NDArrayInfo.NUMPY):
            _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def locvar_jit(rows, cols, ws, arr, res):
    for i, row in enumerate(rows):
        res[row] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    return res**0.5
"""
        elif (ndi == pycd.NDArrayInfo.CUPY):
            _code = r"""
import numba.cuda as nbcuda

@nbcuda.jit(device=True, fastmath=True, opt=True)
def locvar_jit(rows, cols, ws, arr, res):
    for i, row in enumerate(rows):
        res[row] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    return res**0.5
"""
        exec(_code, locals())
        return eval("locvar_jit")

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        res = self._compute_locvar(self._wrow, self._wcol, self._wdata, arr, arr*0.0)
        return pycgspu.convert_arr(res, self._arraymodule, ndi)
        
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        raise NotImplementedError
        
class TotalVariation(pyca.Func):
    r"""
    Graph total variation.
    
    Bases: ``pycsou.abc.operator.Func``
    
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph total variation : :math:`\mathbb{R}^N \rightarrow \mathbb{R}` , is defined as
    
    .. math::
        {S_1 (f) = \sum_{i \in \mathcal{V}}  (\sum_{j \in \mathcal{V}} w_{ij} (f_i - f_j)^2)^{1/2}}
    """
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True
    ):
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        super().__init__(dim = len(self._wdata))
        
        self._dtype = self._wdata[0].dtype
        self._arraymodule = pycd.NDArrayInfo.from_obj(self._wdata)
        self._enable_warnings = bool(enable_warnings)

        self._compute_totvar = self._create_totvar_func(self._arraymodule)
        
    @staticmethod
    def _create_totvar_func(ndi):
        if (ndi == pycd.NDArrayInfo.NUMPY):
            _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def totvar_jit(rows, cols, ws, arr, ys, res):
    for i, row in enumerate(rows):
        ys[row] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    for y in ys:
        res[0] += y**0.5
    return res
"""
        elif (ndi == pycd.NDArrayInfo.CUPY):
            _code = r"""
import numba.cuda as nbcuda

@nbcuda.jit(device=True, fastmath=True, opt=True)
def totvar_jit(rows, cols, ws, arr, ys, res):
    for i, row in enumerate(rows):
        ys[row] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    for y in ys:
        res[0] += y**0.5
    return res
"""
        exec(_code, locals())
        return eval("totvar_jit")
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        xp = self._arraymodule.module()
        res = self._compute_totvar(self._wrow, self._wcol, self._wdata, arr, arr*0.0, xp.array([0.0], dtype=self._dtype))
        return pycgspu.convert_arr(res, self._arraymodule, ndi)
        

class LaplacianQuadratic(pyca.Func):
    r"""
    Graph laplacian quadratic function.
    
    Bases: ``pycsou.abc.operator.Func``
    
    **Mathematical Notes**
    
    Given a graph signal :math:`\mathbf{f} \in \mathbb{R}^N`, where :math:`N = |\mathcal{V}|`, the graph laplacian quadratic function : :math:`\mathbb{R}^N \rightarrow \mathbb{R}` , is defined as
    
    .. math::
        {S_2 (f) = \sum_{i,j \in \mathcal{V}} w_{ij} (f_i - f_j)^2}
    """
    def __init__(
        self,
        W: typ.Union[pyct.NDArray, pyct.SparseArray],
        enable_warnings: bool = True
    ):
        self._wrow, self._wcol, self._wdata = pycgspu.canonical_repr(W)
        super().__init__(dim = len(self._wdata))
        
        self._dtype = self._wdata[0].dtype
        self._arraymodule = pycd.NDArrayInfo.from_obj(self._wdata)
        self._enable_warnings = bool(enable_warnings)

        self._compute_lapquad = self._create_lapquad_func(self._arraymodule)
        
    @staticmethod
    def _create_lapquad_func(ndi):
        if (ndi == pycd.NDArrayInfo.NUMPY):
            _code = r"""
import numba as nb

@nb.jit(nopython=True, fastmath=True)
def lapquad_jit(rows, cols, ws, arr, res):
    for i, row in enumerate(rows):
        res[0] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    return res
"""
        elif (ndi == pycd.NDArrayInfo.CUPY):
            _code = r"""
import numba.cuda as nbcuda

@nbcuda.jit(device=True, fastmath=True, opt=True)
def lapquad_jit(rows, cols, ws, arr, res):
    for i, row in enumerate(rows):
        res[0] += ws[i] * ((arr[row] - arr[cols[i]])**2)
    return res
"""
        exec(_code, locals())
        return eval("lapquad_jit")
    
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: ``pyct.NDArray``
            Input array
        Returns
        -------
        ``pyct.NDArray``
            Output array
        """
        ndi = pycd.NDArrayInfo.from_obj(arr)
        arr = pycgspu.cast_warn(arr, self._dtype, ndi, self._arraymodule, self._enable_warnings)
        xp = self._arraymodule.module()
        res = self._compute_lapquad(self._wrow, self._wcol, self._wdata, arr, xp.array([0.0], dtype=self._dtype))
        return pycgspu.convert_arr(res, self._arraymodule, ndi)
        
        
        
    
