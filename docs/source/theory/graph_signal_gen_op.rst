Generalized Operators for Graph Signals
=======================================

Fundamential operations of signals in classical signal processing can be generalized to the graph signals.

Convolution
-----------
Convolution in classical signal processing for discrete-time signals is defined as:

.. math::
    (f \ast h)_n = \sum_{k=0}^{N-1} f_k h_{n-k}
    
In Fourier domain, it's the multiplication operator:

.. math::
    \widehat{(f \ast h)}_k = \hat{f}_k.\hat{h}_k
    
Since the shifting of the graph signal, :math:`h_{n-k}` is not easy to do so, we can define the convolution of graph signals as the multiplication in graph spectral domain:

.. math::
    \widehat{(f \ast h)}(\lambda_l) = \hat{f}(\lambda_l). \hat{h}(\lambda_l)
    
Taking the inverse graph fourier transform, the convolution equation becomes:

.. math::
    (f \ast h)_i = \sum_{l = 0}^{N-1} \hat{f}(\lambda_l) \hat{h}(\lambda_l) u_l(i)
    
where :math:`u_l(i)` is the :math:`i^\text{th}` element of the eigenvector of graph Laplacian corresponding to the eigenvalue :math:`\lambda_l`.

Translation
-----------
Translation in classical signal processing for continuous-time signals is defined as:

.. math::
    (\mathcal{T}_k f)(t) = f(t-k)
    
It can be also seen as the convolution with :math:`\delta(t-k)`.
    
In Fourier domain, it's the multiplication with eigenfunction corresponding to eigenvalue, i.e. frequency :math:`2\pi k`:

.. math::
    \widehat{(\mathcal{T}_l f)}_k = \hat{f}(w).e^{j k w}

Since the translation operation is not easy to do so in the graph signals, we can define it as the graph convolution with impulse signal:

.. math::
    (\mathcal{T}_k f)_i = \sqrt{N} (f \ast \delta_k)_i
    
Note that the impulse graph signal is defined as:

.. math::
    \delta_k(i) \begin{cases} 1, \:\:\: \text{if} \:\: i=k\\0,\:\:\: \text{o.w.}\end{cases}
    
where the impulse graph signal in the graph spectral domain is:

.. math::
    \hat{\delta}_k(\lambda_l) = u^\ast_l(k)
    
Then, the translation operator can be written as:

.. math::
    (\mathcal{T}_k f)_i = \sqrt{N} \sum_{l=0}^{N-1} \hat{f}(\lambda_l) u^{\ast}_l(k) u_l(i)

.. note::
    :math:`\sqrt{N}` term makes the conversation of the mean of the graph signals:
    
    .. math::
        \sum_{i \in \mathcal{V}} (\mathcal{T}_k f)_i = \sum_{i \in \mathcal{V}} f_i
        
    Proof:

    Using the fact that :math:`\sum_{i \in \mathcal{V}} f_i = \sqrt{N} \hat{f}(0)` and :math:`u_0(i) = \sqrt{N}`, we can make the following derivation:
        
    .. math::
        \sum_{i \in \mathcal{V}} (\mathcal{T}_k f)_i &= \sqrt{N} \widehat{(\mathcal{T}_k f)}(0) \\
        &= N \hat{f}(0) u^\ast_0(k) \\
        &= \sqrt{N} \hat{f}(0) = \sum_{i \in \mathcal{V}} f_i
        
.. note::
    Unlike the translation operator in classical signal processing, it's not isometric:
    
    .. math::
        ||\mathcal{T}_k f||_2 \neq ||f||_2
        
 
Modulation
----------
Modulation in classical signal processing for continuous-time signals is defined as:

.. math::
    (\mathcal{M}_{w_0} f)(t) = e^{j w_0 t} f(t)

where in Fourier domain:

.. math::
    \widehat{(\mathcal{M}_{w_0} f)}(w) = \hat{f}(w-w_0)

It can be also seen as the convolution with :math:`\delta(w-w_0)`, i.e. translation operation in Fourier domain.

The way to generalize the modulation to the graph setting is to define it as the multiplication with eigenfunction in vertex domain:

.. math::
    (\mathcal{M}_k f)_i = \sqrt{N} f_i u_k(i)
    
.. note::
    This generalized modulation is not a translation in the graph spectral domain due to the irregular nature of the spectrum.

Dilation
--------
Dilation in classical signal processing for continuous-time signals is defined as:

.. math::
    (\mathcal{D}_s f)(t) = \frac{1}{s} f(\frac{1}{s})
    
where in Fourier domain it is:

.. math::
    \widehat{(\mathcal{D}_s f)}(w) = \hat{f}(sw)
    
Dilation operation cannot be directly generalized to the graph setting. Scaling in graph spectral domain can be used to generalize this operation:

.. math::
    \widehat{(\mathcal{D}_s f)}(\lambda_l) = \hat{f}(s \lambda_l)
    
.. note::
    Generalized dilation requires the kernel :math:`\hat{f}(.)` to be defined on the entire line, not just on :math:`[0, \lambda_{max}]`.

Downsampling
------------
