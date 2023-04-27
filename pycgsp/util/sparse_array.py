import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

def get_sparse_array_module(x, linalg: bool = False, fallback: pyct.SparseModule = None) -> pyct.SparseModule:
    """
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

