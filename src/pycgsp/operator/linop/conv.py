r"""
Graph convolution operator.

This module provides base classes for polynomial linear operator and graph convolution operator.

.. rubric:: Classes for Graph Convolution

.. autosummary::

    GraphConvolution
"""

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycgsp.util as pycgspu
import pygsp


class _PolyLinOp(pyca.LinOp):
    r"""
    Polynomial Linear Operator :math:`P(L)`.
    
    Bases: ``pycsou.abc.operator.LinOp``
    
    Base class for polynomial operators.
    
    Given a polynomial :math:`P(x) = \sum_{k=0}^N a_k x^k` and a square linear operator :math:`\mathbf{L} : \mathbb{R}^N \rightarrow \mathbb{R}^N`, we define the polynomial linear operator :math:`P(\mathbf{L}) : \mathbb{R}^N \rightarrow \mathbb{R}^N` as:
    
    .. math::
        {P(\mathbf{L}) = \sum_{k=0}^N a_k \mathbf{L}^k},
        
    where :math:`\mathbf{L}^0` is the identity matrix. The adjoint of :math:`P(\mathbf{L})` is given by:
    
    .. math::
        {P(\mathbf{L})^* = \sum_{k=0}^N a_k (\mathbf{L}^*)^k}
        
    * Chebyshev Method:
    
    To lower computational cost, chebyshev approximation for :math:`\mathbf{y} = P(L)(\mathbf{x})` is applied:
    
    .. math::
        {\mathbf{y} = [\bar{\mathbf{x}}_{0} ... \bar{\mathbf{x}}_{N-1}]} a
    
    where :math:`\bar{\mathbf{x}}_{k} = 2 \tilde{\mathbf{L}} \bar{\mathbf{x}}_{k-1} - \bar{\mathbf{x}}_{k-2}` with :math:`\bar{\mathbf{x}}_{0} = \mathbf{x}`, :math:`\bar{\mathbf{x}}_1 = \tilde{\mathbf{L}} \mathbf{x}`.
    
    Here, :math:`\tilde{\mathbf{L}} = 2\mathbf{L}/\lambda_{max} - 1`.
        
    Examples
    --------
    
    >>> import pycsou.abc as pyca
    >>> import pycgsp.linop.conv as pycgspc
    >>> import numpy as np
    >>> L = pyca.LinOp.from_array(A=np.arange(64).reshape(8,8))
    >>> PL = pycgspc.PolyLinOp(LinOp=L, coeffs=np.array([1/2, 2, 1]))
    >>> x = np.arange(8)
    >>> y1 = PL(x)
    >>> y2 = x/2 + 2 * L(x) + (L**2)(x)
    >>> np.allclose(y1, y2)
    True
    
    """
    
    def __init__(self, LinOp: pyca.LinOp, coeffs: pyct.NDArray):
        r"""
        Parameters
        ----------
        LinOp: ``pycsou.abc.operator.LinOp``
            Input linear operator.
        coeffs: ``pyct.NDArray``
            Coefficients of polynomial
        """
        self._LinOp = LinOp
        self._coeffs = coeffs
        self._N = len(coeffs)
        super(_PolyLinOp, self).__init__(shape=LinOp.shape)
        #self._lipschitz = ...
        
    #@pycrt.enforce_precision(i="arr")
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
        # Arraymodules of coeffs and arr should be the same
        y = self._coeffs[0] * arr
        z = arr
        for i in range(1, self._N):
            z = self._LinOp(z)
            y += self._coeffs[i] * z
        return y

            
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
        z = arr
        y = self._coeffs[0] * arr
        for i in range(1, self._N):
            z = self._LinOp.adjoint(z)
            y += self._coeffs[i] * z
        return y
    

def GraphConvolution(
        L: pyca.LinOp,
        kernel=None,
        lmax=None,
        coeffs=None,
        order: int =30,
        method: str = "chebyshev"
    ):
    
    if method == "poly_linop":
        #Lop = pyca.LinOp.from_array(L)
        if coeffs is None:
            raise ValueError("Coeffs must be given")
        PolyLinOp = _PolyLinOp(LinOp=L, coeffs=coeffs)
    else:
        #lmax = L.lipschitz(tight=True)*1.01 # Compute it inside init
        coeffs = pycgspu.compute_cheby_coeff(kernel, lmax, m=order)
        polylinop_coeffs = pycgspu.compute_cheby_polynomial(coeffs)
        PolyLinOp = _PolyLinOp(LinOp=L, coeffs=polylinop_coeffs)
        print(polylinop_coeffs)
    
    return PolyLinOp

    


