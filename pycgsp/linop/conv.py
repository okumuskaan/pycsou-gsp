# #############################################################################
# conv.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Graph convolution operator.
"""

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pygsp

class PolyLinOp(pyca.LinOp):
    """
        Polynomial Linear Operator
    """
    
    def __init__(self, LinOp: pyca.LinOp, coeffs: pyct.NDArray):
        self.LinOp = LinOp
        self.coeffs = coeffs
        #self._lipschitz = ...
        
        super(PolyLinOp, self).__init__(shape=LinOp.shape)
        
    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        y = self.coeffs[0] * arr
        z = arr
        for i in range(1, len(self.coeffs)):
            z = self.LinOp(z)
            y += self.coeffs[i] * z
        return y
            
    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        z = arr
        y = xp.conj(self.coeffs[0]) * arr
        for i in range(1, len(self.coeffs)):
            z = self.LinOp.adjoint(z)
            y += xp.conj(self.coeffs[i]) * z
        return y

class GraphConvolution(PolyLinOp):
    r"""
    Graph convolution.

    Convolve a signal :math:`\mathbf{u}\in\mathbb{C}^N` defined on a graph with a polynomial filter :math:`\mathbf{D}:\mathbb{C}^N\rightarrow \mathbb{C}^N`
    of the form:

    .. math::

       \mathbf{D}=\sum_{k=0}^K \theta_k \mathbf{L}^k,

    where :math:`\mathbf{L}:\mathbb{C}^N\rightarrow \mathbb{C}^N` is the *normalised graph Laplacian* (see [FuncSphere]_ Section 2.3 of Chapter 6).

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pygsp.graphs import RandomRegular
       from pycgsp.linop.conv import GraphConvolution
       np.random.seed(0)

    .. doctest::

       >>> G = RandomRegular(seed=0)
       >>> G.compute_laplacian(lap_type='normalized')
       >>> signal = np.random.binomial(n=1,p=0.2,size=G.N)
       >>> coefficients = np.ones(shape=(3,))
       >>> ConvOp = GraphConvolution(Graph=G, coefficients=coefficients)
       >>> filtered = ConvOp * signal

    .. plot::

       import numpy as np
       from pygsp.graphs import Ring
       from pycgsp.linop.conv import GraphConvolution
       np.random.seed(0)
       G = Ring(N=32, k=2)
       G.compute_laplacian(lap_type='normalized')
       G.set_coordinates(kind='spring')
       signal = np.random.binomial(n=1,p=0.2,size=G.N)
       coefficients = np.ones(3)
       ConvOp = GraphConvolution(Graph=G, coefficients=coefficients)
       e1 = np.zeros(shape=G.N)
       e1[0] = 1
       filter = ConvOp * e1
       filtered = ConvOp * signal
       plt.figure()
       ax=plt.gca()
       G.plot_signal(signal, ax=ax, backend='matplotlib')
       plt.title('Signal')
       plt.axis('equal')
       plt.figure()
       ax=plt.gca()
       G.plot_signal(filter, ax=ax, backend='matplotlib')
       plt.title('Filter')
       plt.axis('equal')
       plt.figure()
       ax=plt.gca()
       G.plot_signal(filtered, ax=ax, backend='matplotlib')
       plt.title('Filtered Signal')
       plt.axis('equal')

    Notes
    -----
    The ``GraphConvolution`` operator is self-adjoint and operates in a matrix-free fashion, as described in Section 4.3, Chapter 7 of  [FuncSphere]_.

    See Also
    --------
    :py:class:`~pycgsp.linop.diff.GraphLaplacian`

    """

    def __init__(self, Graph: pygsp.graphs.Graph, coefficients: pyct.NDArray):
        r"""
        Parameters
        ----------
        Graph: `pygsp.graphs.Graph <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph>`_
            Graph on which the signal is defined, with normalised Laplacian ``Graph.L`` precomputed (see `pygsp.graphs.Graph.compute_laplacian(lap_type='normalized') <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian>`_.
        coefficients: Union[np.ndarray, list, tuple]
            Coefficients :math:`\{\theta_k, \,k=0,\ldots,K\}\subset \mathbb{C}` of the polynomial filter.
        dtype: type
            Type of the entries of the graph filer.

        Raises
        ------
        AttributeError
            If ``Graph.L`` does not exist.
        NotImplementedError
            If ``Graph.lap_type`` is 'combinatorial'.
        """
        self.Graph = Graph
        if Graph.L is None:
            raise AttributeError(r'Please compute the normalised Laplacian of the graph with the routine https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian')
        elif Graph.lap_type != 'normalized':
            raise NotImplementedError(r'Combinatorial graph Laplacians are not supported.')
        else:
            L = self.Graph.L.tocsc()
            Lop = pyca.LinOp.from_array(L)
        self.coefficients = coefficients
        super(GraphConvolution, self).__init__(LinOp=Lop, coeffs=self.coefficients)

