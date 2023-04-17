Graph Signals
=============

Graph Signal Definition
-----------------------

A graph can be defined as follows:

.. math:: \mathcal{G} = \{ \mathcal{V}, \mathcal{E}, \mathbf{W} \}

where

* :math:`\mathcal{V}`: set of vertices with :math:`|\mathcal{V}|=N`
* :math:`\mathcal{E}`: set of edges with :math:`|\mathcal{E}|=N_e`
* :math:`\mathbf{W}`: edge weight matrix with :math:`\mathbf{W} \in \mathbb{R}^{N\times N}`

.. seealso::

   Check graph construction methods.

Given a graph :math:`\mathcal{G}`, a signal can be defined on this graph, which is called *graph signal* as follows:

.. math:: f : \mathcal{V} \rightarrow \mathbb{R}

This graph signal can be represented as a vector:

.. math:: \textbf{f} \in \mathbb{R}^N


**HERE PLOT A GRAPH SIGNAL**




    
Graph Gradient, Divergence, Laplacian
-------------------------------------

**Graph Gradient:**
Graph gradient, :math:`\nabla_{\mathcal{G}} : \mathbb{R}^N \rightarrow \mathbb{R}^{N \times N}` is defined as

.. math:: (\nabla_{\mathcal{G}} f)_{ij} = \sqrt{w_{ij}} (f_i - f_j)

.. note::
    In fact, this kind of gradient operator is called Jacobian operator.


**Graph Divergence:**
Graph divergence, :math:`\text{div}_{\mathcal{G}} : \mathbb{R}^{N \times N} \rightarrow  \mathbb{R}^N` is defined as

.. math:: (\text{div}_{\mathcal{G}} F)_i = \sum_{j \in \mathcal{V}} \sqrt{w_{ij}} F_{ij}


**Graph Laplacian:**
Graph Laplacian, :math:`L : \mathbb{R}^N \rightarrow \mathbb{R}^N` is defined as the divergence of the gradient:

.. math:: (L f)_{i} = (\text{div}_{\mathcal{G}} (\nabla_{\mathcal{G}} f))_i = \sum_{j \in \mathcal{V}} w_{ij} (f_i - f_j)

This linear operator can be represented by matrix :

.. math:: \mathbf{L} = \mathbf{D} - \mathbf{W}

where :math:`\mathbf{D}` is the degree matrix. It's a diagonal matrix whose ith element is :math:`d_i = \sum_{j \in \mathcal{V}} w_{ij}`.

.. note::
    Since :math:`\mathbf{L}` is a real symmetric matrix,  it's positive semidefinite.

This type is called **combinotorial Graph Laplacian**.

We also have **normalized Graph Laplacian**, which is defined as follows:

.. math:: \mathbf{L}^{\text{normalized}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}

**Graph Hessian:**
Graph Hessian, :math:`H : \mathbb{R}^N \rightarrow \mathbb{R}^{N \times N \times N}` is defined as

.. math:: (H f)_{ijk} = \frac{w_{ij}}{2} (f_i - f_j) + \frac{w_{ik}}{2} (f_i - f_k)



Graph Signal Smoothness
-----------------------

* Local variation of a graph signal :math:`f` at vertex :math:`i`:

.. math:: || \nabla_i f ||_2 = \begin{bmatrix} \sum_{e \in \mathcal{E} \text{ s.t. $e=(i,j)$ for some $j \in \mathcal{V}$}} (\left. \frac{\partial f}{\partial e} \right\vert_i)^2 \end{bmatrix}^{\frac{1}{2}}

.. math:: || \nabla_i f ||_2 = \begin{bmatrix} \sum_{j \in \mathcal{N}_i} W_{i,j} [f(j) - f(i)]^2 \end{bmatrix}^{\frac{1}{2}}

* Discrete p-Dirichlet form of :math:`f`:

.. math:: S_p(f) = \frac{1}{p} \sum_{i \in \mathcal{V}} || \nabla_i f ||^p_2 = \frac{1}{p} \sum_{i \in \mathcal{V}} [\sum_{j \in \mathcal{N}_i} W_{i,j} [f(j)-f(i)]^2]^{\frac{p}{2}}

This is for **global smoothness**.

* :math:`p=1`: total variation of the signal wrt graph
* :math:`p=2`:
   
.. math:: S_2(f) = \sum_{(i,j) \in \mathcal{E}} W_{i,j} [f(j) - f(i)]^2 = \textbf{f}^T \mathbf{L} \textbf{f}

Graph Fourier Transform
-----------------------



