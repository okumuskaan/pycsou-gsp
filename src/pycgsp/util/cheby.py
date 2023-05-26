import functools
import numpy as np
import numba as nb

import pycsou.util as pycu
import pycgsp.util.sparse as pycgspsp


def compute_cheby_coeff(kernel, lmax, m=30, N=None, xp=np):
    r"""
    Compute Chebyshev coefficients for a Filterbank.
    Parameters
    ----------
    f : Filter
        Filterbank with at least 1 filter
    m : int
        Maximum order of Chebyshev coeff to compute
        (default = 30)
    N : int
        Grid order used to compute quadrature
        (default = m + 1)
    i : int
        Index of the Filterbank element to compute
        (default = 0)
    Returns
    -------
    c : ndarray
        Matrix of Chebyshev coefficients
    """
        
    if not N:
        N = m + 1

    a_arange = [0, lmax]

    a1 = (a_arange[1] - a_arange[0]) / 2
    a2 = (a_arange[1] + a_arange[0]) / 2
    c = xp.zeros(m + 1)

    tmpN = xp.arange(N)
    num = xp.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * xp.dot(kernel(a1 * num + a2),
                               xp.cos(np.pi * o * (tmpN + 0.5) / N))

    return c
    
def cheby_op(L, N, lmax, c, signal, xp=np):
    r"""
    Chebyshev polynomial of graph Laplacian applied to vector.
    Parameters
    ----------
    L : SparseArray
    c : NDArray or list of ndarrays
        Chebyshev coefficients for a Filter or a Filterbank
    signal : ndarray
        Signal to filter
    Returns
    -------
    r : ndarray
        Result of the filtering
    """
    # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
    #if not isinstance(c, np.ndarray):
    #    c = np.array(c)

    sp = pycgspsp.get_sparse_array_module(L)
    # TODO: check if sp and xp are compatible!

    c = xp.atleast_2d(c)

    Nscales, M = c.shape

    if M < 2:
        raise TypeError("The coefficients have an invalid shape")

    # thanks to that, we can also have 1d signal.
    try:
        Nv = xp.shape(signal)[1]
        r = xp.zeros((N * Nscales, Nv))
    except IndexError:
        r = xp.zeros((N * Nscales))
    
    a_arange = [0, lmax]

    a1 = float(a_arange[1] - a_arange[0]) / 2.
    a2 = float(a_arange[1] + a_arange[0]) / 2.

    twf_old = signal
    twf_cur = 2*(L.dot(signal) - a2 * signal) / a1

    tmpN = xp.arange(N, dtype=int)
    for i in range(Nscales):
        r[tmpN + N*i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

    factor = 2/a1 * (L - a2 * sp.eye(N))
    for k in range(2, M):
        twf_new = factor.dot(twf_cur) - twf_old
        for i in range(Nscales):
            r[tmpN + N*i] += c[i, k] * twf_new

        twf_old = twf_cur
        twf_cur = twf_new

    return r

def _cheby(order, max_order, prev):
    # Given order k, it will return the multipliers for each power
    #             ┌
    #             │ 1                               if  k = 0
    # T_{k}(y) =  ┤ y                               if  k = 1
    #             │ 2y T_{k-1}(y) - T_{k-2}(y)      otherwise
    #             └
    coefs = np.zeros(max_order+1, dtype="int")
    if order == 0:
        coefs[-1] += 1
    elif order == 1:
        coefs[-2] += 1
    else:
        coefs[:-1] += 2 * prev[-1][1:]
        coefs-= prev[-2]
    return coefs


def compute_cheby_polynomial(coefs):
    output = []
    max_order = len(coefs)
    for order in range(max_order):
        res = _cheby(order, max_order, output)
        output.append(res)
    order_coefs = coefs.reshape(-1, 1) * np.array(output)
    return order_coefs.sum(0) # or sum(1) not sure