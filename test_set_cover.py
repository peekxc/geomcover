
# %% 
import numpy as np
from numpy import pi
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from landmark import landmarks
output_notebook()

## Simple dataset with two parts of a spiral
def spiral(N, sigma=1.0):
  theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
  r_a, r_b = 2*theta + pi, -2*theta - pi
  data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T + sigma*np.random.randn(N,2)
  data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T + sigma*np.random.randn(N,2)
  return(np.vstack((data_a, data_b)))

# %% Make an plot data set
S = spiral(N=300, sigma=0.50)
p = figure(width=250, height=250)
p.scatter(*S.T)
show(p)

# %% Construct neighborhood cover
# from tallem.dimred import neighborhood_graph
from set_cover import tangent_neighbor_graph
G, weights, tangents = tangent_neighbor_graph(S, d=1, r=0.75)
# np.all(np.ravel(G.sum(axis=1).flatten()) >= 0)


# %% View individual subset 
plt.scatter(*S.T, c='blue', s=15.5)
for j in range(G.shape[1]):
  plt.scatter(*S[G[:,j].indices,:].T, s=15.5)

# %% Visualize the tangent space + its orthogonal complement
Tx = tangents[0].flatten()
Ty = np.linalg.svd(Tx.reshape((2,1)))[0][:,1]

subset_centered = S[G[:,0].indices,:] - S[0,:]
plt.scatter(*subset_centered.T)
plt.plot([0.0, Tx[0]], [0.0, Tx[1]], c='red')
plt.plot([0.0, Ty[0]], [0.0, Ty[1]], c='green')
plt.gca().set_aspect('equal')

# %% Smallest bounding box with tangent vector as basis 
import matplotlib
from matplotlib.collections import PatchCollection
def subset_bbox(i, S, cover):
  from matplotlib.patches import Polygon
  subset_centered = S[cover[:,i].indices,:] - S[i,:]
  max_proj_x = np.max(np.abs(subset_centered @ np.reshape(Tx, (2, 1))))
  max_proj_y = np.max(np.abs(subset_centered @ np.reshape(Ty, (2, 1))))
  npo, epo = max_proj_x*Tx, max_proj_y*Ty
  return(Polygon([(-npo-epo)+S[i,:], (npo-epo)+S[i,:], (npo+epo)+S[i,:], (-npo+epo)+S[i,:]], True))

plt.scatter(*subset_centered.T)
p = PatchCollection([subset_bbox(0, S, G)], cmap=matplotlib.cm.jet, alpha=0.4)
plt.gca().set_aspect('equal')
plt.gca().add_collection(p)

# %% Plot all subsets
plt.scatter(*S.T, s=1.5)
p = PatchCollection([subset_bbox(i, S, G) for i in range(G.shape[1])], cmap="jet", alpha=0.025)
plt.gca().set_aspect('equal')
plt.gca().add_collection(p)

# %% Tangent weights w/ inner product
from geomstats.geometry.stiefel import StiefelCanonicalMetric
d, D = tangents[0].shape
scm = StiefelCanonicalMetric(d, D)

weights = np.zeros(G.shape[0])
for i in range(G.shape[0]):
  val, ind = G[i,:].nonzero()
  direction = 0.0
  for j in ind:
    direction += 1.0 - np.linalg.norm(scm.inner_product(tangents[i], tangents[j], base_point=np.zeros(shape=(d,D))))
  weights[i] = direction / len(ind)

# %% Plot tangent space estimates + weights 
from matplotlib import cm
from matplotlib.colors import to_hex
from tallem.color import bin_color, rgb_to_hex

col_pal = [to_hex(col) for col in cm.get_cmap('turbo').colors]
col_points = bin_color(weights, col_pal, scaling='linear')

fig = plt.figure(figsize=(4,4), dpi=200)
ax = plt.gca()
s = 1.05
ax.scatter(*S.T, s=8, c=col_points, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
for x, w, T_x, col in zip(S, weights, tangents, col_points):
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  plt.plot(*p.T, c=col, alpha=0.90, linewidth=1.15, zorder=10)
ax.axis('off')
ax.set_aspect('equal')

# %% LP Solution 
from set_cover import wset_cover_LP, wset_cover_greedy, wset_cover_sat


cover_subset, cover_cost = wset_cover_greedy(G, weights)
print(f"Greedy produced a {np.sum(cover_subset)}-set cover with cost {np.sum(weights[cover_subset])}")

cover_subset, cover_cost = wset_cover_LP(G, weights)
print(f"LP produced a {np.sum(cover_subset)}-set cover with cost {np.sum(weights[cover_subset])}")

cover_subset, cover_cost = wset_cover_sat(G, weights)
print(f"MaxSAT produced a {np.sum(cover_subset)}-set cover with cost {np.sum(weights[cover_subset])}")

from pysat.examples.rc2 import RC2
from set_cover.wset_cover import _maxsat_wcnf
wcnf = _maxsat_wcnf(G, weights)
solver = RC2(wcnf)
assignment = np.array(solver.compute())


try: 
  wset_cover_sat(G, weights)
except KeyboardInterrupt:
  print("hello")
finally: 
  x = 1

## Check that we have a valid cover
np.all(G[:,np.flatnonzero(assignment > 0)].sum(axis=1).flatten() > 0)
cover_ind = np.flatnonzero(assignment > 0)
# for a in solver.enumerate(0): 
# 	assignment = np.array(a)
# 	ind = assignment[assignment >= 0]
# 	print(f"MaxSAT produced a {len(ind)}-set cover with weight {np.sum(weights[ind])}")
 

np.array(G[:,cover_subset].sum(axis=1).flatten())


# %% Brute force 
import itertools
def powerset(l):
  for sl in itertools.product(*[[[], [i]] for i in l]):
    yield {j for i in sl for j in i}

for s in powerset(range(5)):
  a = np.array(list(s))
  valid_cover = np.all(G[:,a].sum(axis=1).flatten() > 0)
  if valid_cover:


# %% Plot the weighted set cover estimates
cover_ind = np.flatnonzero(cover_subset)
fig = plt.figure(figsize=(4,4), dpi=200)
ax = plt.gca()
s = 1.50
for i, x in zip(cover_ind, S[cover_ind,:]): # enumerate(X):
  T_x = tangents[i] # (ambient dim x local dim)
  p = np.vstack([x + s*T_x.T, x - s*T_x.T])
  ax.plot(*p.T, c=col_points[i], alpha=0.90, linewidth=1.15, zorder=10)
ax.scatter(*S.T, s=8, c=col_points, alpha=0.55, zorder=20, edgecolor="gray", linewidths=0.50)
ax.axis('off')
ax.set_aspect('equal')

# %%
from tallem.samplers import landmarks
P, _ = landmarks(S, S.shape[0])

w2 = (np.argsort(P)+1)*0.005*weights
cover_subset, cover_cost = wset_cover_LP(G, w2)
print(f"LP produced a {np.sum(cover_subset)}-set cover with cost {np.sum(weights[cover_subset])}")