# %% 
import numpy as np 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from landmark import landmarks
output_notebook(hide_banner=True, verbose=False)

# %% Data set 

## Cosine wave 
x = np.linspace(0, 12*np.pi, (6*16)+1, endpoint=True)
y = np.cos(x)
X = np.c_[x,y]

## Circle  
# x = np.cos(np.linspace(0, 2*np.pi, 32, endpoint=False))
# y = np.sin(np.linspace(0, 2*np.pi, 32, endpoint=False))
# X = np.c_[x,y]

# %% Show cosine wave
p = figure(width=450, height=150)
p.line(x, y, color='blue')
p.scatter(x, y, color='blue', size=3)
show(p)

# %% First step: tangent bundle estimation
from set_cover.csgraph import path_graph, cycle_graph
from scipy.sparse import dia_array
from set_cover.covers import tangent_bundle
from set_cover.plotting import plot_tangent_bundle

## Our manifold will be the simple path graph (+ identity)
n = len(x)
M = path_graph(n, k=5) + dia_array(np.eye(n))
# M = cycle_graph(n, k=5) + dia_array(np.eye(n))
TM = tangent_bundle(M=M, X=X, d=1, centers=X)

## Plot the bundle 
p = plot_tangent_bundle(TM, width = 450, height = 150, x_range = p.x_range, y_range = p.y_range)
# p = plot_tangent_bundle(TM, width = 450, height = 450, x_range = p.x_range, y_range = p.y_range)
show(p)

## Plot the cover 
# p = plot_nerve(M, X, width = 450, height = 150)
# show(p)

# %% Step 2: choose a bundle weighting scheme
from set_cover.covers import bundle_weights
from map2color import map2hex, map2rgb

# TW = bundle_weights(M, TM, method="cosine", reduce=np.mean) # lambda x: np.ptp(x/2.0)
# TW = bundle_weights(M, TM, method="distance", reduce=np.max)
TW = bundle_weights(M, TM, method="angle", reduce=np.mean) # uses Stiefel Canonical Metric 

(x,y),(xs,ys) = plot_tangent_bundle(TM, data=True)
p = figure(width = 450, height = 150)
p.multi_line(xs,ys)
p.scatter(x,y,color=map2hex(TW, 'viridis'), size=5, fill_alpha=1.0)
show(p)


# from geomstats.geometry.stiefel import StiefelCanonicalMetric, Stiefel
# manifold = Stiefel(*TM[0].tangent_vec.shape) # 1-frames in R^2
# metric = StiefelCanonicalMetric(manifold)

# A = M.tocoo()
# weights = np.zeros(len(TM))
# for j in range(len(TM)):
#   tangent_a = TM[j].tangent_vec
#   base_point = TM[j].base_point[:,np.newaxis]
#   neighbor_ind = A.row[A.col == j]
#   print(np.max([metric.inner_product(tangent_a, TM[i].tangent_vec, base_point = tangent_a) for i in neighbor_ind]))
#   weights[j] = (manifold.p * 0.5) - np.mean([metric.inner_product(tangent_a, TM[i].tangent_vec, base_point = tangent_a) for i in neighbor_ind])

# (x,y),(xs,ys) = plot_tangent_bundle(TM, data=True)
# p = figure(width = 450, height = 150)
# p.multi_line(xs,ys)
# p.scatter(x,y,color=map2hex(weights, 'viridis'), fill_alpha=1.0)
# show(p)

# %% Step 3: form the minimal weight set cover 
from set_cover.wset_cover import wset_cover

# cover, cover_weight = wset_cover(M, np.ones(len(TW)), "greedy")
cover, cover_weight = wset_cover(M, max(TW) - TW, "sat")
cover_ind = np.flatnonzero(cover)
# cover_ind = np.array([0,8,16,24])
# assert valid_cover(M, ind=np.flatnonzero(cover))

# from math import ceil 
# # ceil(len(X) / 7)
# cover_ind = np.flatnonzero(np.arange(len(X)) % 7 == 0)
# valid_cover(M, cover_ind)

p = figure(width = 450, height = 150)
p.scatter(*X[cover].T, color='red', size=7, line_color='black')
p.scatter(*X[~cover].T, color='gray', size=4, line_color='gray', fill_alpha=0.50)
show(p)


# from scipy.spatial.distance import cdist 
# base_points = np.array([p for p,v in TM]) # n x D
# tangent_vec = np.array([v.T.flatten() for p,v in TM]) # n x D x d
# A = M.tocoo()




from geomstats.geometry.stiefel import StiefelCanonicalMetric
d, D = TM[0][1].shape

# scm = StiefelCanonicalMetric(d, D)

# weights = np.zeros(G.shape[0])
# for i in range(G.shape[0]):
#   val, ind = G[i,:].nonzero()
#   direction = 0.0
#   for j in ind:
#     direction += 1.0 - np.linalg.norm(scm.inner_product(tangents[i], tangents[j], base_point=np.zeros(shape=(d,D))))
#   weights[i] = direction / len(ind)
  
## Project all points onto tangent plane, measure distance between projected points and original
# sigma = 1.5
# cd = (np.sqrt(2*np.pi) * sigma)**(-1)


  # 
  # tangent_inner_prod =(neighbor_coords - pt).dot(T_y)

    # weights[j] = stat_f(proj_dist)
  # relative_weight = cd * np.exp(-dist_to_basepoint / 2*(sigma**2))
  # proj_coords = np.dot(pt, T_y) # project basepont onto d-tangent plane
  # proj_points = np.array([np.sum(p*T_y, axis=1) for p in proj_coords]) # orthogonal projection in D dimensions
	# proj_dist = np.array([np.sqrt(np.sum(diff**2)) for diff in (pt - proj_points)])
  # weights[i] = np.sum([np.sqrt(np.sum(diff**2)) for diff in (pt - proj_points)]) # np.linalg.norm(centered_pts - proj_points)
	



np.array([3,5], dtype=int)
assert TM[0][1].shape[1] == 1, "Only applies to 1-d tangent bundles"


from scipy.sparse.csgraph import floyd_warshall
A = M.tocoo()
A.data = np.linalg.norm(X[A.row] - X[A.col], axis=1)
D = floyd_warshall(A.todense())

# np.dot(tangent_vec[4][np.newaxis,:], tangent_vec[[3,5]].T) / 2.0
## Between [-1,1]
cos_sim = 1.0 - cdist(tangent_vec[4][np.newaxis,:], tangent_vec[[3,5]], metric='cosine').flatten()


np.ptp(cs)







	# [np.ravel(TM[i][1]) for i in A.row[A.col == j]]
	# ## Span seems pretty decent 
	# mean_cos_sim = lambda j: np.mean(1.0 - cdist(TM[j][1].T, [np.ravel(TM[i][1]) for ind in A.row[A.col == j]], metric="cosine"))
	# TS_alignment = np.array([(cos_sim_span(j)) for j in range(M.shape[1])])
