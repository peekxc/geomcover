# %% 
import numpy as np 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from landmark import landmarks
output_notebook(hide_banner=True, verbose=False)

# %% Data set 
x = np.linspace(0, 12*np.pi, (6*16)+1, endpoint=True)
y = np.cos(x)
X = np.c_[x,y]

# %% Show cosine wave
p = figure(width=450, height=150)
p.line(x, y, color='blue')
p.scatter(x, y, color='blue', size=3)
show(p)

# %% First step: tangent bundle estimation
from set_cover.csgraph import path_graph, cycle_graph
from scipy.sparse import dia_array
from set_cover.covers import tangent_bundle
from set_cover.plotting import plot_tangent_bundle, plot_nerve

## Our manifold will be the simple path graph (+ identity)
n = len(x)
M = path_graph(n, k=5) + dia_array(np.eye(n))
TM = tangent_bundle(M=M, X=X, d=1, centers=X)

## Plot the bundle 
p = plot_tangent_bundle(TM, width = 450, height = 150, x_range = p.x_range, y_range = p.y_range)
show(p)

## Plot the cover 
p = plot_nerve(M, X, width = 450, height = 150)
show(p)

# %% Step 2: choose a bundle weighting scheme
from set_cover.covers import bundle_weights
from map2color import map2hex, map2rgb

# TW = bundle_weights(M, TM, method="cosine", reduce=np.mean) # lambda x: np.ptp(x/2.0)
# TW = bundle_weights(M, TM, method="distance", reduce=np.max)
TW = bundle_weights(M, TM, method="angle", reduce=np.mean) # uses Stiefel Canonical Metric 

(x,y),(xs,ys) = plot_tangent_bundle(TM, data=True)
p = figure(width = 450, height = 150)
p.multi_line(xs,ys)
p.scatter(x,y,color=map2hex(TW, 'viridis'), size=5, fill_alpha=1.0, line_color='gray', line_width=0.50)
show(p)

# %% Step 3: form the minimal weight set cover 
from set_cover.wset_cover import wset_cover

# cover, cover_weight = wset_cover(M, np.ones(len(TW)), "greedy")
cover, cover_weight = wset_cover(M, TW, "sat")
cover_ind = np.flatnonzero(cover)

p = figure(width = 450, height = 150)
p.scatter(*X[cover].T, color='red', size=7, line_color='black')
p.scatter(*X[~cover].T, color='gray', size=4, line_color='gray', fill_alpha=0.50)
show(p)
