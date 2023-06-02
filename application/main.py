from pycgsp.core.plot import myGraphPlot, myGraphPlotSignal
import healpy as hp
import pygsp
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse, spatial, linalg

data_path = "/Users/kaanokumus/Documents/pycsou-gsp/application/data/lambda_WHAM_1_256.fits"

wham_gt = hp.read_map(data_path)
nside = 2*16
wham_gt -=  np.min(wham_gt)
wham_gt +=  1e-9
wham_gt /=  np.max(wham_gt)
wham_gt = np.log10(wham_gt)
wham_gt = hp.pixelfunc.ud_grade(wham_gt, nside)

hp.orthview(wham_gt, cmap="cubehelix")
hp.mollview(wham_gt, cmap="cubehelix")

def hpix_nngraph(hpix_map):
    npix = len(hpix_map)
    nside = hp.npix2nside(npix)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    R = np.stack((x, y, z), axis=-1)
    cols = hp.get_all_neighbours(nside, np.arange(npix)).transpose().reshape(-1)
    cols[cols == -1] = npix - 1
    rows = np.repeat(np.arange(npix), 8, axis=-1).transpose().reshape(-1)

    W = sparse.coo_matrix((cols * 0 + 1, (rows, cols)), shape=(npix, npix))
    extended_row = np.concatenate([W.row, W.col])
    extended_col = np.concatenate([W.col, W.row])
    W.row, W.col = extended_row, extended_col
    W.data = np.concatenate([W.data, W.data])
    W = W.tocsr().tocoo()
    distance = linalg.norm(R[W.row, :] - R[W.col, :], axis=-1)
    rho = np.mean(distance)
    W.data = np.exp(- (distance / rho) ** 2)
    W = W.tocsc()

    return W, R

W, R = hpix_nngraph(wham_gt)
sphere = pygsp.graphs.Graph(W, coords=R)


myGraphPlot(sphere)

print("Everything is OK!")

plt.show()