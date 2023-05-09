import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct

def convert_arr(x, ndi, ndo):
    if (ndi == ndo):
        return x
        
    elif (ndo == pycd.NDArrayInfo.DASK):
        if (ndi == pycd.NDArrayInfo.CUPY):
            x = x.get()
        xp = ndo.module()
        return xp.from_array(x)
    
    elif (ndo == pycd.NDArrayInfo.CUPY):
        if (ndi == pycd.NDArrayInfo.DASK):
            x = x.compute()
        xp = ndo.module()
        return xp.array(x)
    else:
        if (ndi == pycd.NDArrayInfo.DASK):
            return x.compute()
        else:
            return x.get()
