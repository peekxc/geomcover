# %% Imports
import numpy as np
from numpy import pi
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from geomcover.geometry import tangent_bundle
from geomcover.plotting import plot_nerve, plot_tangent_bundle

output_notebook()


# %% Simple dataset with two parts of a spiral
def spiral(N, sigma=1.0):
	theta = np.sqrt(np.random.rand(N)) * 2 * pi  # np.linspace(0,2*pi,100)
	r_a, r_b = 2 * theta + pi, -2 * theta - pi
	data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + sigma * np.random.randn(N, 2)
	data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T + sigma * np.random.randn(N, 2)
	return np.vstack((data_a, data_b))


# %% Data set
X = spiral(N=300, sigma=0.50)
p1 = figure(width=325, height=325, match_aspect=True)
p1.scatter(*X.T, size=4, color="blue", line_color="gray")
show(p1)

# %% First step: tangent bundle estimation
from geomcover.geometry import neighbor_graph_knn

## Our manifold will be the r-neighborhood graph
n = len(X)
# M = neighbor_graph_ball(X, radius=1.00)
M = neighbor_graph_knn(X, k=15, weighted=False)
TM = tangent_bundle(X=X, M=M, d=1, centers=X)

## Plot the bundle
p2 = plot_tangent_bundle(TM, width=325, height=325, match_aspect=True)
show(p2)

## Plot the cover
# p = plot_nerve(M, X, width = 450, height = 150)
# show(p)

# %% Step 2: choose a bundle weighting scheme
from geomcover.geometry import bundle_weights
from map2color import map2hex

# TW = bundle_weights(M, TM, method="cosine", reduce=np.mean)  # lambda x: np.ptp(x/2.0)
# TW = bundle_weights(M, TM, method="distance", reduce=np.max)
TW = bundle_weights(M, TM, method="angle", reduce=np.mean)  # uses Stiefel Canonical Metric

(x, y), (xs, ys) = plot_tangent_bundle(TM, data=True, c=2.0)
p3 = figure(width=350, height=350, match_aspect=True)
p3.multi_line(xs, ys, color=map2hex(TW, "viridis"))
p3.scatter(x, y, color=map2hex(TW, "viridis"), size=3.5, fill_alpha=1.0)
show(p3)


# %% Step 3: form the minimal weight set cover
from geomcover.cover import set_cover

TW_unit = np.ones(len(TW))

# cover, cover_weight = wset_cover(M, TW_unit, "greedy")
# cover, cover_weight = wset_cover(M, TW_unit, "LP", sparsity=0.25)
# cover, cover_weight = wset_cover(M, TW, "LP", sparsity=0.25)
# cover, cover_weight = wset_cover(M, TW, "sat")
# cover, cover_weight = set_cover(M, TW_unit, "ILP")
# cover, cover_weight = wset_cover(M, TW, "sat")
# cover_ind = np.flatnonzero(cover)
# assert valid_cover(M, ind=np.flatnonzero(cover))
cover, cover_weight = set_cover(M, TW, "ILP")

xs_c, ys_c = list(np.array(xs)[cover]), list(np.array(ys)[cover])
p4 = figure(width=350, height=350, match_aspect=True)
p4.scatter(*X[~cover].T, color="gray", size=4, line_color="gray", fill_alpha=0.50)
p4.scatter(*X[cover].T, color="red", size=5, line_color="black")
p4.multi_line(xs_c, ys_c, color="red", line_width=3)
show(p4)

# %%
from bokeh.layouts import row

show(row(p1, p2, p3, p4))


# %% Show nerve simplification
X[cover]

show(plot_nerve(M, X))

# # %% Step 4: Tweak the weights to match what we want
# from scipy.stats import gaussian_kde

# kde = gaussian_kde(X.T)
# density = kde(X.T)
# alignment = bundle_weights(M, TM, method="angle", reduce=np.mean)

# normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
# TW = 0.5 * normalize(alignment) + 0.5 * normalize(density)

# cover, cover_weight = wset_cover(M, TW, "LP")
# cover_ind = np.flatnonzero(cover)

# xs_c, ys_c = list(np.array(xs)[cover]), list(np.array(ys)[cover])
# p = figure(width=350, height=350, match_aspect=True)
# p.scatter(*X[~cover].T, color="gray", size=4, line_color="gray", fill_alpha=0.50)
# p.scatter(*X[cover].T, color="red", size=5, line_color="black")
# p.multi_line(xs_c, ys_c, color="red", line_width=4)
# show(p)
