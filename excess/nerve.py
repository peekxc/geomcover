import bokeh
import numpy as np

from typing import *
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix
from bokeh_nerve import BokehNerveGraph

# B = BokehNerveGraph(cover)
#B.plot()
# B.plot_dynamic(lambda: np.random.uniform(size=(B.G.shape[0], 2)))

# %% 
normalize = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
import pandas as pd
wvs = pd.read_csv("wvs_wave6.csv")
X = np.asanyarray(wvs)[:1500,1:]
X = X - X.mean(axis=0)
X = np.apply_along_axis(normalize, 1, X)

## Construct a PCA-based Mapper
from tallem.dimred import pca
Z = pca(X, 2)
# Z = np.random.uniform(size=(50,2))

d, k = 2, 5
r = 1.5
min_bnds = np.apply_along_axis(np.min, axis=0, arr=Z)
max_bnds = np.apply_along_axis(np.max, axis=0, arr=Z)
box_size = (max_bnds - min_bnds)/k
scaled_box_size = r*box_size

from itertools import product
from array import array
row_indices = array('I')
col_indices = array('I')
data_values = array('f')

eps = 100*np.finfo(float).resolution
center_coords = np.array([np.linspace(min_bnds[i], max_bnds[i], k, endpoint=False) for i in range(d)]).T
index_iterator = product(*[list(range(k)) for di in range(d)])
for j, index in enumerate(index_iterator):
  center = np.array([center_coords[index[di], di] for di in range(d)]) + (box_size/2.0)
  in_box = ~np.zeros(Z.shape[0], dtype=bool)  
  for di in range(d):
    in_box = np.logical_and(in_box, abs(Z[:,di] - center[di]) <= ((scaled_box_size[di]/2.0) + eps))
  row_indices.extend(np.flatnonzero(in_box))
  col_indices.extend(np.repeat(j, np.sum(in_box)))
  data_values.extend(np.linalg.norm(Z[in_box,:]-center, axis=1))

from scipy.sparse import csc_matrix, coo_matrix
cover_image = coo_matrix((data_values, (row_indices, col_indices)), shape=(Z.shape[0], k**d))
cover_image = cover_image.tocsc()

cover = cover_image.copy()
# plt.scatter(*Z.T)
# indices = [cover_image[:,j].indices for j in range(cover_image.shape[1])]
# indices = np.hstack(indices)
# plt.scatter(*Z[indices,:].T, c='black')
# for j, index in enumerate(product(*[list(range(k)) for di in range(d)])):
#   center = np.array([center_coords[index[di], di] for di in range(d)])+ (box_size/2.0)
#   plt.scatter(*center)
# plt.gca().set_aspect('equal')

np.sum([len(cover_image[:,j].indices) for j in range(cover_image.shape[1])])
max_ind = np.argmax([len(cover_image[:,j].indices) for j in range(cover_image.shape[1])])

## Do SL clustering on the preimages 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from kneed import KneeLocator

for j in range(cover_image.shape[1]):
  L = linkage(pdist(X[cover_image[:,j].indices,:]), method='single')
  LD = L[:,2]
  kneedle = KneeLocator(np.array(range(len(LD))), LD, S=0.05, curve="convex", direction="increasing")
  cl = fcluster(L, kneedle.knee_y, criterion='distance')


L = linkage(pdist(X[cover_image[:,max_ind].indices,:]), method='single')
linkage_dist = L[:,2]
linkage_dist[linkage_dist > eps]
fcluster(L, 0.50, criterion='distance')
plt.plot(linkage_dist[linkage_dist > eps])

subset_full = X[cover_image[:,max_ind].indices,:]
subset_low = cmds(subset_full,2)
plt.scatter(*subset_low.T)
ind_sep = np.logical_or(subset_low[:,0] <= -0.25, subset_low[:,0] >= 0.25)
plt.scatter(*subset_low[ind_sep,:].T)

L = linkage(pdist(subset_low[ind_sep,:]), method='single')
linkage_dist = L[:,2]
plt.plot(linkage_dist)


LD = linkage_dist
kneedle = KneeLocator(np.array(range(len(LD))), LD, S=0.05, curve="convex", direction="increasing")
plt.plot(LD)
plt.vlines(x=kneedle.knee, ymin=0, ymax=1, colors='red', ls=':', lw=2)

cl = fcluster(L, kneedle.knee_y, criterion='distance')


cl = fcluster(L, t=2, criterion='maxclust_monocrit', monocrit=LD)
np.unique(cl)

plt.scatter(*subset_low[ind_sep,:].T, c=cl)

import matplotlib.pyplot as plt
plt.scatter(*Z.T)