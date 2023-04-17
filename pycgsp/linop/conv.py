r"""
Graph convolution operator.

This module provides base classes for polynomial linear operator and graph convolution operator.

.. rubric:: Classes for Graph Convolution

.. autosummary::

    PolyLinOp
    GraphConvolution
"""

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycgsp.util as pycgspu
import pygsp

class PolyLinOp(pyca.LinOp):
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
    
    def __init__(self, LinOp: pyca.LinOp, coeffs: pyct.NDArray, method="exact"):
        r"""
        Parameters
        ----------
        LinOp: ``pycsou.abc.operator.LinOp``
            Input linear operator.
        coeffs: ``pyct.NDArray``
            Coefficients of polynomial
        method: str
            Method to implement polynomial linear operator, either ``exact`` or ``chebyshev``. Default: ``exact``.
        """
        self.LinOp = LinOp
        self.coeffs = coeffs
        self._N = len(self.coeffs)
        if method!="chebyhev" or method!="exact":
            raise ValueError("Method should be either 'exact' or 'chebyshev'.")
        self._method = method
        super(PolyLinOp, self).__init__(shape=LinOp.shape)
        #self._lipschitz = ...
        
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
        if self._method == "exact":
            y = self.coeffs[0] * arr
            z = arr
            for i in range(1, len(self.coeffs)):
                z = self.LinOp(z)
                y += self.coeffs[i] * z
            return y
        else:
            # Chebyshev method
            raise NotImplementedError("Chebyshev Method Not Supported Yet")
            c = pycgspu.compute_cheby_coeff(self, m=self._N-1)

            
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
        xp = pycu.get_array_module(arr)
        if self._method == "exact":
            z = arr
            y = xp.conj(self.coeffs[0]) * arr
            for i in range(1, len(self.coeffs)):
                z = self.LinOp.adjoint(z)
                y += xp.conj(self.coeffs[i]) * z
            return y
        else:
            # Chebyshev method
            raise NotImplementedError("Chebyshev Method Not Supported Yet")

class GraphConvolution(PolyLinOp):
    r"""
    Graph Convolution Operator.

    Bases: ``pycgsp.linop.conv.PolyLinOp``
    
    Given an input graph signal :math:`\mathbf{f_{in}} \in \mathbb{R}^N`, output graph signal :math:`\mathbf{f_{out}} \in \mathbb{R}^N` through a filtering kernel :math:`\mathbf{g}` can be defined as:
    
    .. math::
        {\mathbf{f_{out}} = \hat{g}(L) \mathbf{f_{in}}}

    where :math:`\hat{g}(L) = U \begin{bmatrix} \hat{g}(\lambda_0) & & 0 \\ & ... & \\ 0 & & \hat{g}(\lambda_{N-1}) \\ \end{bmatrix} U^T`.
    
    Here, eigen-decomposition of graph laplacian is defined as :math:`L = U \Lambda U^T` with eigenvalues :math:`\{\lambda_0, ..., \lambda_{N-1}\}`.
    
    Denote filtering kernel as :math:`\hat{g}(\lambda_k) = \theta_k`, so :math:`\hat{g}(\Lambda) = \text{diag}\{\theta_0, ..., \theta_{N-1}\}`.
    
    To make it localized, polynomial approximation is applied:
    
    .. math::
        {g_\theta(\Lambda) = \sum_{k=0}^{K-1} \theta_k \Lambda^k}
    
    where :math:`\theta \in \mathbb{R}^K` is a vector of polynomial coefficients. Then, :math:`(g_\theta(L))_{ij} = \sum_k \theta_k (L^k)_{ij}`.
    
    
    TODO: Add kernel input to implement chebyshev approximation

    """

    def __init__(self, Graph, coefficients: pyct.NDArray, method: str = "chebyshev"):
        r"""
        Parameters
        ----------
        Graph: (``pycgsp.core.graph.Graph``, ``pygsp.graphs.Graph``)
            Graph object.
        coefficients: ``pyct.NDArray``
            Coefficients of graph convolution.
        method: ``str``
            Method to compute graph convolution, either ``exact`` or ``chebyshev``.
        """
        if Graph.L is None:
            raise AttributeError(r'Please compute the normalised Laplacian of the graph with the routine https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian')
        elif Graph.lap_type != 'normalized':
            raise NotImplementedError(r'Combinatorial graph Laplacians are not supported.')
        else:
            L = self.Graph.L.tocsc()
            Lop = pyca.LinOp.from_array(L)
        super(GraphConvolution, self).__init__(LinOp=Lop, coeffs=coefficients, method=method)

