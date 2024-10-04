"""Cosine curve example - geomcover package."""

# %% Imports
import numpy as np
import scipy as sp
from geomcover.csgraph import path_graph
from geomcover.geometry import tangent_bundle
from geomcover.plotting import plot_nerve, plot_tangent_bundle
from geomcover.geometry import bundle_weights

from geomcover.cover import set_cover
from map2color import map2hex
from bokeh.io import output_notebook
from bokeh.plotting import figure, show

output_notebook(hide_banner=True, verbose=False)

# %% Data set
x = np.linspace(0, 12 * np.pi, (6 * 16) + 1, endpoint=True)
y = np.cos(x)
X = np.c_[x, y]
p = figure(width=450, height=150)
p.line(x, y, color="blue")
p.scatter(x, y, color="blue", size=3)
show(p)

# %% First step: tangent bundle estimation

## Our manifold will be the simple path graph (+ identity)
n = len(x)
M = path_graph(n, k=5) + sp.sparse.dia_array(np.eye(n))
TM = tangent_bundle(X=X, M=M, d=1, centers=X)

## Plot the bundle
p = plot_tangent_bundle(TM, width=450, height=150, x_range=p.x_range, y_range=p.y_range)
show(p)

## Plot the cover
p = plot_nerve(M, X, width=450, height=150)
show(p)

# %% TODO: make tangentPairs classesthat admit projection methods
# from geomstats.learning.preprocessing import ToTangentSpace
# from geomstats.learning.pca import TangentPCA

# ToTangentSpace

# %% Step 2: choose a bundle weighting scheme
TW = bundle_weights(M, TM, method="cosine", reduce=np.mean)  # lambda x: np.ptp(x/2.0)
# TW = bundle_weights(M, TM, method="distance", reduce=np.max)
# TW = bundle_weights(M, TM, method="angle", reduce=np.mean) # uses Stiefel Canonical Metric

(x, y), (xs, ys) = plot_tangent_bundle(TM, data=True)
p = figure(width=450, height=150)
p.multi_line(xs, ys)
p.scatter(x, y, color=map2hex(TW, "viridis"), size=5, fill_alpha=1.0, line_color="gray", line_width=0.50)
show(p)

# %% Step 3: form the minimal weight set cover
# cover, cover_weight = wset_cover(M, np.ones(len(TW)), "greedy")
cover, cover_weight = set_cover(M, TW, "ILP")
cover_ind = np.flatnonzero(cover)

p = figure(width=450, height=150)
p.scatter(*X[cover].T, color="red", size=7, line_color="black")
p.scatter(*X[~cover].T, color="gray", size=4, line_color="gray", fill_alpha=0.50)
show(p)
