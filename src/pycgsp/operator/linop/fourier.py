r"""
Graph fourier transform operators.

This module provides graph fourier and graph inverse fourier transforms.

.. rubric:: Classes for Fourier Transforms

.. autosummary::

    GraphFourier
    GraphInvFourier
    
"""

import typing as typ

import pycsou.abc as pyca
import pycsou.util.ptype as pyct
import pycsou.runtime as pycrt

import pycgsp.operator.linop.diff as pycgspd

__all__ = [
    "GraphFourier",
    "GraphInvFourier",
]




class GraphFourier(pyca.NormalOp):
    r"""
    Graph Fourier Transform.
    """
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray],
                 lap_type: str = "combinatorial"):
        r"""
        """
        
        self._N = W.shape[0]
        
        LapOp =  pycgspd.GraphLaplacian(W, lap_type=lap_type)
        
        self._U = LapOp.eigvals(k=self._N)
        
        self._lap_type = lap_type
        super().__init__(shape=(self._N, self._N))
      
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        
        """
        return self._U.T.dot(arr)
        
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        return self._U.dot(arr)
        
    
class GraphInvFourier(pyca.NormalOp):
    r"""
    Graph Inverse Fourier Transform.
    """
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray],
                 lap_type: str = "combinatorial"):
        r"""
        """
        
        self._N = W.shape[0]
        
        LapOp =  pycgspd.GraphLaplacian(W, lap_type=lap_type)
        
        self._U = LapOp.eigvals(k=self._N)
        
        self._lap_type = lap_type
        super().__init__(shape=(self._N, self._N))
      
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
    
        return self._U.dot(arr)
        
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        return self._U.T.dot(arr)
        
    
        
