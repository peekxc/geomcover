# %% Imports
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from geomcover.csgraph import cycle_graph
from geomcover.geometry import tangent_bundle
from geomcover.plotting import plot_tangent_bundle
from scipy.sparse import dia_array

output_notebook(hide_banner=True, verbose=False)

# %% Circle
x = np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False))
y = np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))
X = np.c_[x, y]

# %% Plot
p1 = figure(width=325, height=325, match_aspect=True)
p1.line(np.append(x, x[0]), np.append(y, y[0]), color="blue", line_width=2)
p1.scatter(x, y, color="blue", size=5)
show(p1)

## Our manifold will be the simple cycle graph (+ identity)
n = len(x)
M = cycle_graph(n, k=7) + dia_array(np.eye(n))
TM = tangent_bundle(M=M, X=X, d=1, centers=X)

# %% Plot the bundle
p2 = plot_tangent_bundle(TM, width=325, height=325, match_aspect=True)
show(p2)

# %% Plot the cover
p3 = plot_nerve(M, X, width=325, height=325, match_aspect=True)
show(p3)

# %% Step 2: choose a bundle weighting scheme
from geomcover.geometry import bundle_weights
from map2color import map2hex

# TW = bundle_weights(M, TM, method="cosine", reduce=np.mean)  # lambda x: np.ptp(x/2.0)
# TW = bundle_weights(M, TM, method="distance", reduce=np.max)
TW = bundle_weights(M, TM, method="angle", reduce=np.mean)  # uses Stiefel Canonical Metric

(x, y), (xs, ys) = plot_tangent_bundle(TM, data=True)
p4 = figure(width=325, height=325, match_aspect=True)
p4.multi_line(xs, ys)
p4.scatter(x, y, color=map2hex(TW, "viridis"), size=5, fill_alpha=1.0)
show(p4)

# %% Step 3: form the minimal weight set cover
from geomcover.cover import set_cover

# cover, cover_weight = wset_cover(M, np.ones(len(TW)), "greedy")
cover, cover_weight = set_cover(M, TW, "ILP")
cover_ind = np.flatnonzero(cover)

p5 = figure(width=325, height=325, match_aspect=True, title="")
p5.scatter(*X[cover].T, color="red", size=7, line_color="black")
p5.scatter(*X[~cover].T, color="gray", size=4, line_color="gray", fill_alpha=0.50)
show(p5)


# %%
from bokeh.layouts import row

show(row(p1, p2, p3, p4, p5))
