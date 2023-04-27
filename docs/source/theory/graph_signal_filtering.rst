Graph Signal Filtering
======================

Filtering on graph signal and its efficient implementation should be studied carefully.

Graph Signal Filtering Formula
------------------------------
In classical signal processing, the filtering of shift-invariant linear systems is equivalent to the convolution of the input signal with impulse response of the system.

For the graph setting, the filtering is defined as the graph convolution of input signal with system kernel. This definition of filtering is called as **spectral filtering** since the graph convolution (put link to graph conv here) is defined in the graph spectral domain:

.. math::
    f_{out}(\lambda_l) = \sum_{l=0}^{N-1} \hat{f}_{in}(\lambda_l) \hat{h}(\lambda_l) u_l(i)
    
This equation can be represented as a matrix vector multiplication equation:

.. math::
    \mathbf{f}_{out} = \hat{h}(\mathbf{L})\mathbf{f}_{in}
    
where :math:`\mathbf{f}_{out}, \mathbf{f}_{out} \in \mathbb{R}^N` and :math:`\hat{h}(\mathbf{L}) = \mathbf{U} \hat{h}(\Lambda) \mathbf{U}^T` with

.. math::
    \hat{h}(\mathbf{\Lambda}) = \begin{bmatrix} \hat{h}(\lambda_0) & & 0 \\ & \ddots & \\ 0 & & \hat{h}(\lambda_{N-1}) \end{bmatrix}
    
* This formula is non-localized in vertex domain.

* Computational complexity of this implementation is in :math:`O(N^2)`.


Polynomial Approximation
------------------------

If polynomial approximation is applied to the kernel in graph spectral domain:

.. math::
    \hat{h}(\lambda_l) \approx \sum_{k=0}^K \theta_k \lambda_l^k

then the filtering becomes as follows:

.. math::
    f_{out}(i) &= \sum_{l=0}^{N-1} \hat{f}_{in}(\lambda_l) \hat{h}(\lambda_l) u_l(i) \\
    &= \sum_{j=1}^N \sum_{l=0}^{N-1} f_{in}(j) u_l^\ast (j) \sum_{k=0}^K \theta_k \lambda_l^k u_l(i) \\
    &= \sum_{j=1}^N f_{in}(j) \sum_{k=0}^K \theta_k (\mathbf{L}^k)_{ij}
    
Thus, we have the following approximation with same formula as above:

.. math::
    \hat{h}(\mathbf{L}) \approx \hat{h}_{\theta}(\mathbf{L}) = \sum_{k=0}^K \theta_k \mathbf{L}^k
    
Here, :math:`(\mathbf{L}^k)_{ij}` is nonzero if there exist a path from vertex :math:`i` to vertex :math:`j` with length :math:`k`. Thus, given :math:`d_{\mathcal{G}}(i,j)` is the geodesic distance, i.e. shortest path length from vertex :math:`i` to vertex :math:`j`, we have the following lemma:

.. math::
    d_{\mathcal{G}}(i,j) > k \:\:\: \Rightarrow \:\:\: (\mathbf{L}^k)_{ij} = 0

* It results in :math:`K`-hop linear **localized** transform:

.. math::
    f_{out}(i) = \sum_{j \in \mathcal{N}(i,K) \cup \{i\}} b_{ij} f_{in}(j)
    
where :math:`b_{ij} = \sum_{k=d_\mathcal{G}(i,j)}^K \theta_k (\mathbf{L}^k)_{ij}`.

* Computational efficiency of this implementation is still in :math:`O(N^2)`.

Chebyshev Approximation
-----------------------
In order to decrease the computational cost of the implementation, chebyshev approximation could be used:

.. math::
    f_{out} = \hat{h}_\theta(\mathbf{L})f_{in} = \bar{\mathbf{f}}_{in}^T \theta \\
    \bar{\mathbf{f}}_{in, k} = 2 \tilde{L} \bar{\mathbf{f}}_{in, k-1} - \bar{\mathbf{f}}_{in, k-2}
    
with :math:`\bar{\mathbf{f}}_{in, 0} = f_{in}`, :math:`\bar{\mathbf{f}}_{in, 1} = \tilde{L} f_{in}` and :math:`\tilde{L} = \frac{L}{\lambda_{max}/2} - 1`.

* Entire filtering operation costs :math:`O(K N_e)`, which less than :math:`O(N^2)` for most of the case particularly for graphs with a sparse adjacency matrix.
    
