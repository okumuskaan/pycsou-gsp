import typing as typ
import warnings

import pycgsp.util as pycgspu

import pycsou.util as pycu
import pycsou.util.ptype as pyct
import pycsou.util.deps as pycd
import pycsou.runtime as pycrt
import pycsou.util.warning as pycuw


def sparse_to_rcd(arr: pyct.SparseArray) -> tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]:
    r"""
    Converts sparse weight matrix to (row, col, data) arrays format
    
    Input sparse matrix is firstly converted to **coo matrix**. Then, (row, col,data) arrays are returned by using the built-in attributes. Input can be either a SCIPY, PYDATA or CUPY sparse matrix.
    
    Parameters
    ----------
    arr: ``pyct.SparseArray``
        Sparse weighted adjacency matrix.
    
    Returns
    -------
    ``tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]``
        (Row, Column, Data) array tuple. Length of each data is two times number of edges.
    """
    spi = pycd.SparseArrayInfo.from_obj(arr)
    if (spi == pycd.SparseArrayInfo.SCIPY_SPARSE) or (spi == pycd.SparseArrayInfo.CUPY_SPARSE):
        sp = spi.module()
        if not sp.isspmatrix_coo(arr):
            arr = arr.tocoo()
        row = arr.row
        col = arr.col
        data = arr.data
    elif (spi == pycd.SparseArrayInfo.PYDATA_SPARSE):
        arr.asformat("coo")
        row, col = arr.coords
        data = arr.data
    else:
        raise ValueError("Unknown input for sparse array.")
    return (row, col, data)




def canonical_repr(W: typ.Union[pyct.NDArray, pyct.SparseArray], symmetric_matrix: bool = True) -> tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]:
    """
    Converts weight matrix W to (row, col, data) arrays.
    
    If the matrix is sparse array, then ``sparse_to_rcd`` function is called.
    If the matrix is array, then the nonzero method and access by index are applied. For the DASK array, it's converted to NUMPY array.
    As a result, (wrow, wcol, wdata) array tuple is returned.
    
    Parameters
    ----------
    arr: ``pyct.NDArray`` || ``pyct.SparseArray``
        Weighted adjacency matrix.
    
    Returns
    -------
    ``tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]``
        (Row, Column, Data) array tuple.
    """
    if (type(W) in pycd.supported_array_types()):
        # TODO: DASK ARRAYS CONVERTED TO NUMPY ARRAYS
        W = pycu.compute(W, traverse=True)
        _wrow, _wcol = W.nonzero()
        _wdata = W[_wrow, _wcol]
    else:
        _wrow, _wcol, _wdata = sparse_to_rcd(W)
    
    if symmetric_matrix:
    # Choose only the upper triangular
        inds = _wcol<=_wrow
        _wrow = _wrow[inds]
        _wcol = _wcol[inds]
        _wdata = _wdata[inds]
    
    #_wdata = pycrt.coerce(_wdata)

    return (_wrow, _wcol, _wdata)



def cast_warn(arr: pyct.NDArray, _dtype, _ndi, _arraymodule, _enable_warnings) -> pyct.NDArray:
    r"""
    Casts input array to backend data type.
    
    If casting is applied, the warning is given in case _enable_warnings is True.
    
    Parameters
    ----------
    arr: ``pyct.NDArray``
        Input array.
    _dtype: ``Data Type``
        Backend data type.
    _enable_warnings: ``bool``
        Default: True
    
    Returns
    -------
    ``pyct.NDArray``
        Casted array.
    """
    if arr.dtype == _dtype:
        out = arr
    else:
        if _enable_warnings:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)
        out = arr.astype(dtype=self._dtype)
    if _ndi != _arraymodule:
        if _enable_warnings:
            msg = "Computation is not performed at the requested arraymodule, as it's not compatible with the backend arraymodule."
            warnings.warn(msg, pycuw.BackendWarning)
        out = pycgspu.convert_arr(out, _ndi, _arraymodule)
    return out

