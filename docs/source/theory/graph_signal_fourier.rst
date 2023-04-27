Graph Fourier Transform
=======================
Using the graph spectral theory, Fourier transform for graph signals can be defined and studied.


Review of Graph Spectral Theory
-------------------------------
For combinatorial graph Laplacian, the eigen-decomposition of Laplacian matrix is defined as:

.. math::
    L = U \Lambda U^T
    
where :math:`\Lambda = \text{diag}(\lambda_0, ..., \lambda_{N-1})`.

Let us assume without loss of generality the eigenvalues are sorted as :math:`0 = \lambda_0 \leq ... \leq \lambda_{N-1} = \lambda_{max}`.


Graph Fourier Transform
-----------------------
In classical signal processing of continuous-time signals, Laplacian operator is defined as :math:`\Delta f(t) = \frac{\partial^2 f(t)}{\partial t^2}`. Then, the eigenfunction of the Laplacian operator is complex exponential function: :math:`\Delta e^{jwt} = -w^2 e^{jwt}`. Then, continous-time Fourier transform is defined as:

.. math::
    \hat{f}(w) = <f(t), e^{jwt}> = \int_{-\infty}^\infty f(t) e^{-jwt}
    
Similarly, the graph Fourier transform can be defined as the projection of signal into eigenvectors of graph Laplacian:

.. math::
    \hat{f}(\lambda_l) = < f(i), u_l(i)> = \sum_{i=1}^N f(i) u_l^\ast (i)
    
In matrix vector formula, this can be written as:

.. math::
    \hat{\mathbf{f}} = \mathbf{U}^T \mathbf{f}
    
Inverse graph Fourier transform can be defined as:

.. math::
    f(i) = \sum_{l=0}^{N-1} \hat{f}(\lambda_l) u_l(i)
    
In matrix vector formula, this can be written as:

.. math::
    \mathbf{f} = \mathbf{U} \hat{\mathbf{f}}

Properties of GFT
-----------------
Eigenvalues of graph Laplacian gives a notion of **frequency**. The less the eigenvalue is, the smoother the graph signal we observe. For example, for :math:`\lambda_0 = 0`, we have :math:`u_0(i)=\frac{1}{\sqrt{N}}, \:\: \forall i \in \mathcal{V}` a constant signal, which is the smoothest one. Using this fact, we also have the following relation:

.. math::
    \hat{f}(0) &= \sum_{i=1}^N f(i) u^\ast_0(i) \\
    &= \frac{1}{\sqrt{N}} \sum_{i = 1}^N f(i)

which is the DC component of the graph signal.

If we define impulse function in vertex domain, then its graph fourier transform would be eigenfunction:

.. math::
    \delta_k(i) = \begin{cases} 1, \:\:\: \text{if} \: i=k \\ 0, \:\:\: \text{o.w.}\end{cases} \:\:\: \leftrightarrow \:\:\: \hat{\delta}_k(\lambda_l) = u_l(k)

Any low-pass, band-pass and high-pass graph signals can be created in graph spectral domain. Let us give an example for a low-pass kernel:

**Heat Kernel:**
The kernel in graph spectral domain is exponential filter with rate :math:`\tau`, which is as follows:

.. math::
    \hat{f}(\lambda_l) = e^{-j \tau \lambda_l}

This have low-pass filter characteristic. Then, if inverse graph Fourier transform is applied, the signal in vertex domain could be obtained.

