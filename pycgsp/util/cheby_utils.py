import functools
import numpy as np

def filterbank_handler(func):

    # Preserve documentation of func.
    @functools.wraps(func)

    def inner(f, *args, **kwargs):

        if 'i' in kwargs:
            return func(f, *args, **kwargs)

        elif f.Nf <= 1:
            return func(f, *args, **kwargs)

        else:
            output = []
            for i in range(f.Nf):
                output.append(func(f, *args, i=i, **kwargs))
            return output

    return inner


@filterbank_handler
def compute_cheby_coeff(f, lmax, m=30, N=None):
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
    c = np.zeros(m + 1)

    tmpN = np.arange(N)
    num = np.cos(np.pi * (tmpN + 0.5) / N)
    for o in range(m + 1):
        c[o] = 2. / N * np.dot(f._kernels[i](a1 * num + a2),
                               np.cos(np.pi * o * (tmpN + 0.5) / N))

    return c
