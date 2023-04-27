Graph Signal Calculus
=====================

Graph Signals and Hilbert Spaces
--------------------------------

A finite weighted graph can be defined as:

.. math:: \mathcal{G} = \{ \mathcal{V}, \mathcal{E}, \mathbf{W} \}

where

* :math:`\mathcal{V} = \{1,...,N\}`: set of vertices with :math:`|\mathcal{V}|=N`
* :math:`\mathcal{E} = \{(i,j) : (i,j) \in \mathcal{V} \times \mathcal{V}, w_{ij}\neq 0\}`: set of edges with :math:`|\mathcal{E}|=N_e` and :math:`\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}`
* :math:`\mathbf{W} = \{w_{ij}\}`: edge weight matrix with :math:`\mathbf{W} \in \mathbb{R}^{N\times N}`

Note that for an undirected graph, :math:`w_{ij} = w_{ji}`.

**Space of Real Signals on Vertex Space:**

Generic real functions on vertex space can be defined as

.. math::
    f : \mathcal{V} \rightarrow \mathbb{R}

Such signals are called as **graph signals**. Then, Hilbert space of such signals can be defined as

.. math::
    \mathcal{H}(\mathcal{V}) = \{ f: \mathcal{V} \rightarrow \mathbb{R} \}

Since :math:`|\mathcal{V}| = N`, :math:`\mathcal{H}(\mathcal{V}) \cong \mathbb{R}^N`.

Inner product of :math:`\mathcal{H}(\mathcal{V})`:

.. math::
    <f, g>_{\mathcal{H}(\mathcal{V})} = \sum_{i \in \mathcal{V}} f_i g_i, \:\:\:\:\:\: \forall f,g \in \mathcal{H}(\mathcal{V})
    
:math:`\ell_p`-norm of :math:`\mathcal{H}(\mathcal{V})`:

.. math::
    ||f||_p = (\sum_{i \in \mathcal{V}} |f_i|^p)^{1/p}, \:\:\:\:\:\: 1 \leq p < \infty
    
.. math::
    ||f||_\infty = \max_{i \in \mathcal{V}} |f_i|
    
Note that :math:`\ell_2`-norm is induced by the inner product.

**Space of Real Signals on Edge Space:**

Generic real functions on edge space can be defined as

.. math::
    F : \mathcal{E} \rightarrow \mathbb{R}
    
Such signals are called as **graph edge signals**. Then, Hilbert space of such signals can be defined as

.. math::
    \mathcal{H}(\mathcal{E}) = \{ F: \mathcal{E} \rightarrow \mathbb{R} \}
    
Since :math:`|\mathcal{E}| = N_e`, :math:`\mathcal{H}(\mathcal{E}) \cong \mathbb{R}^{N_e}`.

Inner product of :math:`\mathcal{H}(\mathcal{E})`:

.. math::
    <F, G>_{\mathcal{H}(\mathcal{E})} = \sum_{(i,j) \in \mathcal{E}} F_{ij} G_{ij}, \:\:\:\:\:\: \forall F,G \in \mathcal{H}(\mathcal{E})
    
:math:`\ell_p`-norm of :math:`\mathcal{H}(\mathcal{E})`:

.. math::
    ||F||_p = (\sum_{(i,j) \in \mathcal{E}} |F_{ij}|^p)^{1/p}, \:\:\:\:\:\: 1 \leq p < \infty
    
.. math::
    ||F||_\infty = \max_{(i,j) \in \mathcal{E}} |F_{ij}|
    
Note that :math:`\ell_2`-norm is induced by the inner product.


Differential Graph Operators
----------------------------

**Graph Gradient:**
Graph gradient :math:`\nabla : \mathcal{H}(\mathcal{V}) \rightarrow \mathcal{H}(\mathcal{E})` is defined as

.. math:: (\nabla f)_{ij} = \sqrt{w_{ij}} (f_i - f_j)

In fact, this kind of gradient operator is called Jacobian operator.
    
Note that for an undirected graph, :math:`(\nabla f)_{ij} = - (\nabla f)_{ji}`.

**Graph Divergence:**
Graph divergence :math:`\text{div} : \mathcal{H}(\mathcal{E}) \rightarrow  \mathcal{H}(\mathcal{V})` is defined as

.. math:: (\text{div} F)_i = \sum_{j \in \mathcal{V}} \sqrt{w_{ij}} (F_{ij} - F_{ji})

.. note::
    For an anti-symmetric graph edge signal, i.e. :math:`F_{ij} = - F_{ji}`, the graph divergence becomes:
    
    .. math::
        (\text{div} F)_i = \sum_{j \in \mathcal{V}} 2\sqrt{w_{ij}} F_{ij}


.. note::
    Graph divergence is the adjoint operator of graph gradient.
    
    Proof:
    
    .. math::
        <\nabla f, G>_{\mathcal{H}(\mathcal{E})} &= \sum_{(i,j) \in \mathcal{E}} (\nabla f)_{ij} G_{ij} \\
            &= \sum_{(i,j) \in \mathcal{E}} \sqrt{w_{ij}}(f_i - f_j) G_{ij} \\
            &= \sum_{(i,j) \in \mathcal{E}} \sqrt{w_{ij}}f_i(G_{ij} - G_{ji}) \\
            &= \sum_{i \in \mathcal{V}} f_i \sum_{j \in \mathcal{V}} \sqrt{w_{ij}}(G_{ij} - G_{ji}) \\
            &= \sum_{i \in \mathcal{V}}f_i (\text{div} G)_i = <f, \text{div} G>_{\mathcal{H}(\mathcal{V})}
            
            






**Graph Laplacian:**
Graph Laplacian :math:`L : \mathcal{H}(\mathcal{V}) \rightarrow \mathcal{H}(\mathcal{V})` is defined as the graph divergence of the graph gradient, :math:`L = \text{div}\nabla`:

.. math::
    (L f)_{i} = (\text{div} (\nabla f))_i = \sum_{j \in \mathcal{V}} w_{ij} (f_i - f_j)

This linear operator can be represented by matrix :

.. math:: \mathbf{L} = \mathbf{D} - \mathbf{W}

where :math:`\mathbf{D}` is the degree matrix. It's a diagonal matrix whose ith element is :math:`d_i = \sum_{j \in \mathcal{V}} w_{ij}`.

.. note::
    Since :math:`\mathbf{L}` is a symmetric matrix with :math:`\mathbf{f}^T \mathbf{L} \mathbf{f} \geq 0`, :math:`\forall \mathbf{f} \in \mathbb{R}^N`, :math:`\mathbf{L}` is a positive semidefinite matrix.
    
    Proof of :math:`\mathbf{f}^T \mathbf{L} \mathbf{f} \geq 0`, :math:`\forall \mathbf{f} \in \mathbb{R}^N`:
    
    .. math::
        \mathbf{f}^T \mathbf{L} \mathbf{f} = \sum_{(i,j) \in \mathcal{E}} \mathbf{f}^T \mathbf{L}_{ij} \mathbf{f} = \sum_{(i,j) \in \mathcal{E}} w_{ij}(f_i - f_j)^2 \geq 0




This type of graph laplacian is called as **combinotorial**.

We can also defined **normalized** graph laplacian as:

.. math:: \mathbf{L}^{\text{normalized}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}

This is equivalent to:

.. math::
    (L^{\text{normalized}}f)_i =  \frac{1}{\sqrt{d_i}} \sum_{i \in \mathcal{V}} w_{ij} (\frac{f_i}{\sqrt{d_i}} - \frac{f_j}{\sqrt{d_j}})
    
Another definition of graph laplacian is **asymmetric** graph laplacian which can be constructed by random walk matrix :math:`\mathbf{P} = \mathbf{D}^{-1}\mathbf{W}` as:

.. math::
    \mathbf{L}_a = \mathbf{I} - \mathbf{P}
    
    
.. list-table:: Comparison of Graph Laplacians
   :header-rows: 1
   :stub-columns: 1
   :align: center

   * -
     - Combinatorial
     - Normalized
     - Asymmetric
   * - Matrix
     - :math:`\mathbf{D} - \mathbf{W}`
     - :math:`\mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}`
     - :math:`\mathbf{I} - \mathbf{D}^{-1}\mathbf{W}`
   * - Eigenvalues
     - :math:`0=\lambda_0 \leq ... \leq \lambda_{N-1} = \lambda_{max}`
     - :math:`0=\tilde{\lambda}_0 \leq ... \leq \tilde{\lambda}_{N-1} = 2`
     - :math:`0=\tilde{\lambda}_0 \leq ... \leq \tilde{\lambda}_{N-1} = 2`
   * - Eigenvectors
     - :math:`\{u_i\}_{i=0}^N` with :math:`u_0 = \mathbf{1} / \sqrt{N}`
     - :math:`\{\tilde{u}_i\}_{i=0}^N` with :math:`u_0` not constant
     - :math:`\{\frac{1}{\sqrt{d_i}}\tilde{u}_i\}_{i=0}^N`
     
     
     

**Graph Hessian:**
Graph Hessian, :math:`H : \mathbb{R}^N \rightarrow \mathbb{R}^{N \times N \times N}` is defined as

.. math:: (H f)_{ijk} = \frac{w_{ij}}{2} (f_i - f_j) + \frac{w_{ik}}{2} (f_i - f_k)

.. note::
    Graph hessian is the trace of graph laplacian.
    
    Proof:
    
    .. math::
        \text{tr}(Hf)_i &= \sum_{j,k \in \mathcal{V}: \: j=k} (Hf)_{ijk} \\
        &= \sum_{j,k \in \mathcal{V}: \: j=k} \frac{w_{ij}}{2} (f_i - f_j) + \frac{w_{ik}}{2} (f_i - f_k) \\
        &= \sum_{j \in \mathcal{V}} w_{ij} (f_i - f_j) = (Lf)_i \\


Graph Signal Smoothness
-----------------------

* Local variation of a graph signal :math:`f` at vertex :math:`i`:

.. math:: || \nabla_i f ||_2 = || (\nabla f)_{i.} ||_2 = \begin{bmatrix} \sum_{j: (i,j) \in \mathcal{E}} (\nabla f)_{ij}^2 \end{bmatrix}^{\frac{1}{2}}

.. math:: || \nabla_i f ||_2 = \begin{bmatrix} \sum_{j: (i,j) \in \mathcal{E}} w_{ij} (f_i - f_j)^2 \end{bmatrix}^{\frac{1}{2}}

This can be used to measure **local** smoothness.

* Discrete p-Dirichlet form of :math:`f`:

.. math:: S_p(f) = \frac{1}{p} \sum_{i \in \mathcal{V}} || \nabla_i f ||^p_2 = \frac{1}{p} \sum_{i \in \mathcal{V}} [\sum_{j \in \mathcal{N}_i} w_{ij} (f_i-f_j)^2]^{\frac{p}{2}}

This can be used to measure **global** smoothness.

* :math:`p=1`: Total variation of the graph signal

.. math:: S_1(f) = \sum_{i \in \mathcal{V}} || \nabla_i f ||_2

* :math:`p=2`: Graph Laplacian quadratic form
   
.. math:: S_2(f) = \sum_{(i,j) \in \mathcal{E}} w_{ij} (f_i - f_j)^2 = \textbf{f}^T \mathbf{L} \textbf{f}

.. note::
    Graph Laplacian quadratic form is equivalent to :math:`\ell^2`-norm square in :math:`\mathcal{H}(\mathcal{E})`:
    
    .. math::
        S_2(f) = || \nabla f ||^2_2
    
    This norm can also be seen as a norm in :math:`\mathcal{H}(\mathcal{V})`:

    .. math::
        S_2(f) = ||\mathbf{L}^{1/2} \textbf{f}||^2_2
        

