# #############################################################################
# diff.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Graph differential operators.
"""

import numpy as np
import pygsp

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pygsp
from pycsou.operator.linop.base import IdentityOp
from conv import PolyLinOp, GraphConvolution

class GraphLaplacian(pyca.LinOp):
    r"""
    Graph Laplacian.

    Normalised graph Laplacian for signals defined on graphs.

    Examples
    --------

    .. plot::

       import numpy as np
       from pygsp.graphs import Ring
       from pycgsp.linop.diff import GraphLaplacian
       np.random.seed(1)
       G = Ring(N=32, k=4)
       G.compute_laplacian(lap_type='normalized')
       G.set_coordinates(kind='spring')
       x = np.arange(G.N)
       signal = np.piecewise(x, [x < G.N//3, (x >= G.N//3) * (x< 2 * G.N//3), x>=2 * G.N//3], [lambda x: -x, lambda x: 3 * x - 4 * G.N//3, lambda x: -0.5 * x + G.N])
       Lap = GraphLaplacian(Graph=G)
       lap_sig = Lap * signal
       plt.figure()
       ax=plt.gca()
       G.plot_signal(signal, ax=ax, backend='matplotlib')
       plt.title('Signal')
       plt.axis('equal')
       plt.figure()
       plt.plot(signal)
       plt.title('Signal')
       plt.figure()
       ax=plt.gca()
       G.plot_signal(lap_sig, ax=ax, backend='matplotlib')
       plt.title('Laplacian of signal')
       plt.axis('equal')
       plt.figure()
       plt.plot(-lap_sig)
       plt.title('Laplacian of signal')

    Notes
    -----
    For undirected graphs, the normalized graph Laplacian is defined as

    .. math:: \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2},

    where :math:`\mathbf{I}` is the identity matrix, :math:`\mathbf{W}` is the weighted adjacency matrix and :math:`\mathbf{D}` the
    weighted degree matrix.

    For directed graphs, the Laplacians are built from a symmetrized
    version of the weighted adjacency matrix that is the average of the
    weighted adjacency matrix and its transpose. As the Laplacian is
    defined as the divergence of the gradient, it is not affected by the
    orientation of the edges.

    For both Laplacians, the diagonal entries corresponding to disconnected
    nodes (i.e., nodes with degree zero) are set to zero.

    The ``GraphLaplacian`` operator is self-adjoint.

    See Also
    --------
    :py:class:`~pycgsp.linop.diff.GraphGradient`, :py:func:`~pycgsp.linop.diff.GeneralisedGraphLaplacian`
    :py:class:`~pycgsp.linop.conv.GraphConvolution`

    """

    def __init__(self, Graph: pygsp.graphs.Graph, dtype: type=np.float):
        r"""
        Parameters
        ----------
        Graph: `pygsp.graphs.Graph <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph>`_
            Graph on which the signal is defined, with normalised Laplacian ``Graph.L`` precomputed (see `pygsp.graphs.Graph.compute_laplacian(lap_type='normalized') <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian>`_.
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
            raise AttributeError(
                r'Please compute the normalised Laplacian of the graph with the routine https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian')
        elif Graph.lap_type != 'normalized':
            raise NotImplementedError(r'Combinatorial graph Laplacians are not supported.')
        else:
            self.L = self.Graph.L.tocsc()
            
        super(GraphLaplacian, self).__init__(shape=self.Graph.W.shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.L.dot(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self(arr) # since it's self-adjoint


class GraphGradient(pyca.LinOp):
    r"""
    Graph gradient.

    Gradient operator for signals defined on graphs.

    Examples
    --------

    .. testsetup::

       import numpy as np
       from pygsp.graphs import Ring
       from pycgsp.linop.diff import GraphLaplacian, GraphGradient
       np.random.seed(1)

    .. doctest::

       >>> G = Ring(N=32, k=4)
       >>> G.compute_laplacian(lap_type='normalized')
       >>> G.compute_differential_operator()
       >>> G.set_coordinates(kind='spring')
       >>> x = np.arange(G.N)
       >>> signal = np.piecewise(x, [x < G.N//3, (x >= G.N//3) * (x< 2 * G.N//3), x>=2 * G.N//3], [lambda x: -x, lambda x: 3 * x - 4 * G.N//3, lambda x: -0.5 * x + G.N])
       >>> Lap = GraphLaplacian(Graph=G)
       >>> Grad = GraphGradient(Graph=G)
       >>> lap_sig = Lap * signal
       >>> lap_sig2 = Grad.adjoint(Grad(signal))
       >>> np.allclose(lap_sig, lap_sig2)
       True

    Notes
    -----
    The adjoint of the ``GraphGradient`` operator is called the graph divergence operator.

    Warnings
    --------
    In the newest version of PyGSP (> 0.5.1) the convention is changed: ``Graph.D`` is the divergence operator and
    ``Graph.D.transpose()`` the gradient (see routine `Graph.compute_differential_operator <https://pygsp.readthedocs.io/en/latest/reference/graphs.html#pygsp.graphs.Graph.compute_differential_operator>`_). The code should be adapted when this new version is released.

    See Also
    --------
    :py:class:`~pycgsp.linop.diff.GraphLaplacian`, :py:func:`~pycgsp.linop.diff.GeneralisedGraphLaplacian`

    """

    def __init__(self, Graph: pygsp.graphs.Graph):
        r"""
        Parameters
        ----------
        Graph: `pygsp.graphs.Graph <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph>`_
            Graph on which the signal is defined, with differential operator ``Graph.D`` precomputed (see `pygsp.graphs.Graph.compute_differential_operator() <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_differential_operator>`_.
        dtype: type
            Type of the entries of the graph filer.

        Raises
        ------
        AttributeError
            If ``Graph.D`` does not exist.
        """
        self.Graph = Graph
        if Graph.D is None:
            raise AttributeError(
                r'Please compute the differential operator of the graph with the routine https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_differential_operator')
        else:
            self.D = self.Graph.D.tocsc()
        super(GraphGradient, self).__init__(shape=self.Graph.W.shape)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.D.dot(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self.D.conj().transpose().dot(arr)


def GeneralisedGraphLaplacian(Graph: pygsp.graphs.Graph, kind: str = 'iterated', **kwargs):
    r"""
    Generalised graph Laplacian operator.

    Generalised Laplacian operator signals defined on graphs.

    Parameters
    ----------
    Graph: `pygsp.graphs.Graph <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph>`_
        Graph on which the signal is defined, with normalised Laplacian ``Graph.L`` precomputed (see `pygsp.graphs.Graph.compute_laplacian(lap_type='normalized') <https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian>`_.
    dtype: type
        Type of the entries of the graph filer.
    kind: str
        Type of generalised differential operator (``'iterated'``, ``'sobolev'``, ``'polynomial'``).
        Depending on the cases, the ``GeneralisedLaplacian`` operator is defined as follows:

        * ``'iterated'``: :math:`\mathscr{D}=\mathbf{L}^N`,
        * ``'sobolev'``: :math:`\mathscr{D}=(\alpha^2 \mathrm{Id}-\mathbf{L})^N`, with :math:`\alpha\in\mathbb{R}`,
        * ``'polynomial'``: :math:`\mathscr{D}=\sum_{n=0}^N \alpha_n \mathbf{L}^n`,  with :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`,

        where :math:`\mathbf{L}` is the :py:class:`~pycgsp.linop.diff.GraphLaplacian` operator.
    kwargs: Any
        Additional arguments depending on the value of ``kind``:

        * ``'iterated'``: ``kwargs={order: int}`` where ``order`` defines the exponent :math:`N`.
        * ``'sobolev'``: ``kwargs={order: int, constant: float}`` where ``order`` defines the exponent :math:`N` and ``constant`` the scalar :math:`\alpha\in\mathbb{R}`.
        * ``'polynomial'``: ``kwargs={coeffs: Union[np.ndarray, list, tuple]}`` where ``coeffs`` is an array containing the coefficients :math:`\{\alpha_0,\ldots,\alpha_N\} \subset\mathbb{R}`.

    Raises
    ------
    AttributeError
        If ``Graph.L`` does not exist.
    NotImplementedError
        If ``Graph.lap_type`` is 'combinatorial'.
    NotImplementedError
        If ``kind`` is not 'iterated', 'sobolev' or 'polynomial'.

    Examples
    --------

    .. plot::

       import numpy as np
       from pygsp.graphs import Ring
       from pycgsp.linop.diff import GeneralisedGraphLaplacian
       np.random.seed(1)
       G = Ring(N=32, k=4)
       G.compute_laplacian(lap_type='normalized')
       G.set_coordinates(kind='spring')
       x = np.arange(G.N)
       signal = np.piecewise(x, [x < G.N//3, (x >= G.N//3) * (x< 2 * G.N//3), x>=2 * G.N//3], [lambda x: -x, lambda x: 3 * x - 4 * G.N//3, lambda x: -0.5 * x + G.N])
       Dop = GeneralisedGraphLaplacian(Graph=G, kind='polynomial', coeffs=[1,-1,2])
       gen_lap = Dop * signal
       plt.figure()
       ax=plt.gca()
       G.plot_signal(signal, ax=ax, backend='matplotlib')
       plt.title('Signal')
       plt.axis('equal')
       plt.figure()
       ax=plt.gca()
       G.plot_signal(gen_lap, ax=ax, backend='matplotlib')
       plt.title('Generalized Laplacian of signal')
       plt.axis('equal')


    Notes
    -----
    The ``GeneralisedGraphLaplacian`` operator is self-adjoint.

    See Also
    --------
    :py:class:`~pycgsp.linop.diff.GraphLaplacian`


    """
    if Graph.L is None:
        raise AttributeError(
            r'Please compute the normalised Laplacian of the graph with the routine https://pygsp.readthedocs.io/en/stable/reference/graphs.html#pygsp.graphs.Graph.compute_laplacian')
    elif Graph.lap_type != 'normalized":
        raise NotImplementedError(r'Combinatorial graph Laplacians are not supported.')
    else:
        L = Graph.L.tocsc()
        LapOp = pyca.LinOp.from_array(L)
        
    if kind == "iterated":
        N = kwargs['order']
        Dgen = LapOp ** N
    elif kind == "sobolev":
        I = IdentityOp(size=LapOp.shape[0] * LapOp.shape[1])
        alpha = kwargs['constant']
        N = kwargs['order']
        Dgen = ((alpha ** 2) * I - LapOp) ** N
    elif kind == "polynomial":
        coeffs = kwargs["coeffs"]
        Dgen = PolyLinOp(LinOp=LapOp, coeffs=coeffs)
    else:
        raise NotImplementedError(
            'Supported generalised Laplacian types are: iterated, sobolev, polynomial.')
    return Dgen
