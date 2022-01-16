from tallem.samplers import landmarks
import numpy as np
from numpy import pi

def spiral(N, sigma=1.0):
  theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)
  r_a, r_b = 2*theta + pi, -2*theta - pi
  data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T + sigma*np.random.randn(N,2)
  data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T + sigma*np.random.randn(N,2)
  return(np.vstack((data_a, data_b)))


# %% Make an plot data set
import matplotlib.pyplot as plt
S = spiral(N=300, sigma=0.50)
plt.scatter(*S.T)


# %% Construct neighborhood cover
from tallem.dimred import neighborhood_graph
from covers import neighborhood_cover
cover, weights, tangents = neighborhood_cover(S, d=1, k=15)

## Tangent weights w/ inner product
from geomstats.geometry.stiefel import StiefelCanonicalMetric
d, D = tangents[0].shape
scm = StiefelCanonicalMetric(d, D)

weights = np.zeros(cover.shape[0])
for i in range(cover.shape[0]):
  val, ind = cover[i,:].nonzero()
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
from set_cover import wset_cover_LP, wset_cover_greedy, wset_cover_greedy_naive
cover_ind, cover_cost = wset_cover_LP(cover, weights)
cover_ind, cover_cost = wset_cover_greedy(cover, 1.0/weights)

# %% Plot the weighted set cover estimates
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
