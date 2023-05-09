import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

def get_sparse_array_module(x, linalg: bool = False, fallback: pyct.SparseModule = None) -> pyct.SparseModule:
    r"""
    Get the sparse array namespace corresponding to a given object.
    
    Parameters
    ----------
    x: object
        Any object compatible with the interface of SciPy.Sparse arrays.
    linalg: bool
        Boolean data that determines whether sparse or sparse.linalg module is returned
        Default: False
    fallback: pyct.SparseModule
        Fallback module if `x` is not a SciPy.Sparse-like array.
        Default behaviour: raise error if fallback used.
    Returns
    -------
    namespace: pyct.SparseModule
        The namespace to use to manipulate `x`, or `fallback` if provided.
    """
    
    def infer_api(y, la):
        try:
            return pycd.SparseArrayInfo.from_obj(y).module(linalg=la)
        except ValueError:
            return None
            
    if (xp := infer_api(x, linalg)) is not None:
        return xp
    elif fallback is not None:
        return fallback
    else:
        raise ValueError(f"Could not infer sparse array module for {type(x)}.")


def sparse_to_rcd(arr: pyct.SparseArray) -> tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]:
    r"""
    Converts sparse weight matrix to numpy (row, col, data)
    
    Here, input sparse matrix is firstly converted to **coo matrix**. Then, (row, col,data) arrays are returned by using the attributes. Input can be either SCIPY, PYDATA or CUPY sparse matrix.
    
    Parameters
    ----------
    arr: ``pyct.SparseArray``
        Sparse matrix.
    
    Returns
    -------
    ``tuple[pyct.NDArray, pyct.NDArray, pyct.NDArray]``
        (Row, Column, Data) array tuple.
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
