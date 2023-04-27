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

class LocalVariation(pyca.SquareOp):
    r"""
    """
    
    def __init__(self, W: typ.Union[pyct.NDArray, pyct.SparseArray]):
        r"""
        """
        
        self._N = W.shape[0]
        
        super().__init__(shape=(self._N , self._N))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        pass
        
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        """
        pass
        
class TotalVariation(pyca.Func):
    r"""
    """
    pass

class LaplacianQuadratic(pyca.Func):
    r"""
    """
    pass
        
    
