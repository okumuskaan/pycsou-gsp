Graph Signal Filtering
======================

Filtering on graph signal and its efficient implementation should be studied carefully.

Graph Signal Filtering Formula
------------------------------
In classical signal processing, the filtering of shift-invariant linear systems is equivalent to the convolution of the input signal with the impulse response of the system.

For the graph setting, the filtering is defined as the graph convolution of input signal with the kernel of the system. This definition of filtering is called as **spectral filtering** since the graph convolution (put link to graph conv here) is defined in the graph spectral domain.

Given the kernel in the spectral domain :math:`g`, we can write the graph convolution operator as:

.. math::
    f_{out}(\lambda_l) = \sum_{l=0}^{N-1} \hat{f}_{in}(\lambda_l) g(\lambda_l) u_l(i)
    
This equation can be represented as a matrix vector multiplication:

.. math::
    \mathbf{f}_{out} = g(\mathbf{L})\mathbf{f}_{in}
    
where :math:`\mathbf{f}_{in}, \mathbf{f}_{out} \in \mathbb{R}^N` and :math:`g(\mathbf{L}) = \mathbf{U} g(\Lambda) \mathbf{U}^T \in \mathbb{R}^{N\times N}` with

.. math::
    g(\mathbf{\Lambda}) = \begin{bmatrix} g(\lambda_0) & & 0 \\ & \ddots & \\ 0 & & g(\lambda_{N-1}) \end{bmatrix}
    
.. note::
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

.. note::
    * It results in :math:`K`-hop linear **localized** transform:

    .. math::
        f_{out}(i) = \sum_{j \in \mathcal{N}(i,K) \cup \{i\}} b_{ij} f_{in}(j)
        
    where :math:`b_{ij} = \sum_{k=d_\mathcal{G}(i,j)}^K \theta_k (\mathbf{L}^k)_{ij}`.

    * Computational efficiency of this implementation is still in :math:`O(N^2)`.

Chebyshev Approximation
-----------------------
This approximation can be done using the chebyshev polynomials.

Chebyshev polynomials :math:`T_k : [-1,1] \rightarrow [-1,1]` can be described as:

.. math::
    T_k(x) = \cos (k \arccos (x))

Another description of Chebyshev polynomials can be defined by its recurrence relation:

.. math::
    T_k(x) = \begin{cases}1, & k=0\\ x, & k=1\\ 2xT_{k-1}(x) - T_{k-2}(x), & k \geq 2 \end{cases}

These Chebyshev polynomials form an orthogonal basis for :math:`\mathcal{L}^2 ([-1,1], dx/\sqrt{1-x^2})`
Every function :math:`g` on :math:`[-1,1]` that is square integrable  with respect to the measure :math:`dx/\sqrt{1-y^2}` can be represented as:

.. math::
    h(x) = \frac{1}{2}c_0 + \sum_{k=1}^\infty c_k T_k (x)
    
where :math:`\{c_k\}_{k=0,1,2,...}` is a sequence of Chebyshev coefficients.

For graph filtering, input range is :math:`[0, \lambda_{max}]`. Shifting and scaling can be applied to Chebyshev polynomials to approximate the kernel :math:`g(\lambda_l)` via the transformation: :math:`\frac{\lambda_{max}}{2}(x+1)`. Then, the kernel can be represented as follows:

.. math::
    g(x) = \frac{1}{2} c_0 + \sum_{k=1}^\infty c_k \overline{T}_k (x), \:\:\: \forall x \in [0, \lambda_{max}]

where :math:`\overline{T}_k (x) = T_k(x/\alpha - 1)`, :math:`\alpha = \frac{\lambda_{max}}{2}` and

.. math::
    c_k = \frac{2}{\pi} \int_0^{\pi} \cos (k \theta) g(\alpha (\cos \theta + 1)) d \theta
    
Recurrence relation becomes:

.. math::
    \overline{T}_k(x) = \begin{cases} 1 & k = 0 \\ x/\alpha - 1 & k = 1\\ \frac{2}{\alpha} (x - \alpha) \overline{T}_{k-1}(x) - \overline{T}_{k-2}(x) & k \geq 2\end{cases}

Then, for input :math:`\mathbf{L} \in \mathbb{R}^{N \times N}`, we have

.. math::
    g(\mathbf{L}) = \frac{1}{2} c_0 \mathbf{I} + \sum_{k=1}^\infty c_k \overline{T}_k (\mathbf{L})
    
where

.. math::
    \overline{T}_k(\mathbf{L}) = \begin{cases} \mathbf{I}, & k=0 \\ \mathbf{L}/\alpha - \mathbf{I}, & k=1 \\ \frac{2}{\alpha} (\mathbf{L} - \alpha \mathbf{I})\overline{T}_{k-1}(\mathbf{L}) - \overline{T}_{k-2}(\mathbf{L}), \\ k \geq 2 \end{cases}

Choosing the order of Chebyshev approximation as :math:`M`:

.. math::
    f_{out} = g(\mathbf{L}) f_{in} \approx \frac{1}{2} c_0 f_{in} + \sum_{k=1}^M c_k \overline{T}_k(\mathbf{L}) f_{in}

.. note::
    * Computational cost is in :math:`O(M N_e)`, which is much lower than :math:`O(N^2)` for sparse weighted adjacency matrix :math:`\mathbf{W}`.
