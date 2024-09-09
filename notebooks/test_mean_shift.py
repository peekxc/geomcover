import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.palettes import Sunset8
from scipy.ndimage import shift
from scipy.stats import gaussian_kde, multivariate_normal, norm

output_notebook()
# multivariate_normal.pdf(X, mean=[0,0], cov=np.eye(2))
X = np.loadtxt("https://raw.githubusercontent.com/mattnedrich/MeanShift_py/master/data.csv", delimiter=",")
K = gaussian_kde(X.T)

# %% show the test data + the contours
p = figure(width=300, height=300)
p.scatter(*X.T)
show(p)

(x_min, y_min), (x_max, y_max) = np.min(X, axis=0), np.max(X, axis=0)
x_delta, y_delta = 0.10 * np.ptp(X[:, 0]), 0.10 * np.ptp(X[:, 1])
xp = np.linspace(x_min - x_delta, x_max + x_delta, 80)
yp = np.linspace(y_min - y_delta, y_max + y_delta, 80)
xg, yg = np.meshgrid(xp, yp)
Z = K.evaluate(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
levels = np.linspace(np.min(Z), np.max(Z), 9)

p = figure(width=550, height=400, match_aspect=True)
contour_renderer = p.contour(xg, yg, Z, levels, fill_alpha=0.0, fill_color=Sunset8, line_color=Sunset8, line_width=3)
colorbar = contour_renderer.construct_color_bar()
p.add_layout(colorbar, "right")
p.scatter(*X.T, fill_alpha=K.evaluate(X.T) / np.max(K.evaluate(X.T)))
show(p)

# %% Perform the mean shift
from geomcover.cluster import mean_shift, assign_clusters, MultivariateNormalZeroMean
from collections import Counter

shift_points = mean_shift(X, bandwidth=K.factor)
cl = assign_clusters(shift_points)

# %%
from map2color import map2hex

# p = figure(width=300, height=300)
p.scatter(*X.T, color=map2hex(cl))
p.scatter(*shift_points.T, color=map2hex(cl), size=8, line_color="gray")
show(p)

# %%
KDE = gaussian_kde(X.T, bw_method=1.0)
cov = Covariance.from_precision(KDE.inv_cov, KDE.covariance)
n, d = X.shape
MVN = MultivariateNormalZeroMean((n, n, d), cov)
# MVN(X)


# import timeit
# X = np.random.uniform(size=(1500, 2))
# n, d = X.shape
# KDE = gaussian_kde(X.T, bw_method=1.0)
# MVN = MultivariateNormalZeroMean(cov)
# delta = X[:, np.newaxis, :] - X[:150, :][np.newaxis, :, :]
# timeit.timeit(lambda: MVN(delta), number=150)
# timeit.timeit(lambda: multivariate_normal.pdf(delta, mean=mu, cov=cov), number=150)


# %% Ensure kDE matches
from scipy.stats import Covariance, multivariate_normal
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import gaussian_kde

# X = np.random.uniform(size=(1500, 2), low=0, high=1)
# mu = np.zeros(2, dtype=np.float64)
# bw = np.repeat(0.25, len(mu))
# cov = Covariance.from_diagonal(bw)

# KDE = gaussian_kde(X.T, bw_method=0.20)
# cov = Covariance.from_precision(KDE.inv_cov, KDE.covariance)

# delta = X[:, np.newaxis, :] - X[np.newaxis, :, :]
# dens1 = np.sum(multivariate_normal.pdf(delta, mean=mu, cov=cov) / len(X), axis=1)
# dens2 = KDE.evaluate(X.T)
# assert np.allclose(dens1, dens2)

# import timeit

# timeit.timeit(lambda: np.sum(multivariate_normal.pdf(delta, mean=mu, cov=cov), axis=1) / len(X), number=10)
# timeit.timeit(lambda: KDE.evaluate(X.T), number=10)

# assert isinstance(kernel, Callable), "Kernel must a be a callable function."

# %% Test two larger gaussians

X1 = multivariate_normal.rvs(size=1500, mean=[-5, 0], cov=1)
X2 = multivariate_normal.rvs(size=1500, mean=[+5, 0], cov=np.diag([1, 2]))
X = np.vstack([X1, X2])

## Blurred mean shift takes 3.7s here, compared to regular MS taking 14.5s
## About 4x faster!
## custom mvn code takes 14.1s, compared to 19.2 w/ scipy
S = X[np.random.choice(range(len(X)), size=500, replace=False)]
SP = mean_shift(X, S, maxiter=200, batch=25)
cl = assign_clusters(SP, atol=0.5)


KDE = gaussian_kde(X.T)
density = KDE.evaluate(X.T)
## Straified samplign approach
# np.quantile(density, np.linspace(0, 1, 10))
cuts = np.linspace(np.min(density), np.max(density), 10)
binned = np.digitize(density, cuts)
bin_counts = np.bincount(binned)
n_samples = 100
(n_samples / len(X)) * bin_counts
# np.random.choice(range(len(X)), replace=False, p=bin_counts/np.sum(bin_counts))

np.digitize(density, bins=10)

p = figure(width=300, height=300)
p.scatter(*X.T, color=map2hex(cl, "turbo"))
# p.scatter(*SP.T, color="red", line_color="gray")
show(p)

from scipy.cluster.hierarchy import fcluster

L = single(pdist(SP))
cl = fcluster(L, 2, criterion="maxclust")


gaussian_kde(X.T)

# multivariate_normal.pdf(X[:5], mean=[0, 0], cov=1)
## covariance object 1.8s -> 1.6s
from line_profiler import LineProfiler
from geomcover.cluster import batch_shift_kernel

profile = LineProfiler()
profile.add_function(mean_shift)
profile.add_function(batch_shift_kernel)
profile.add_function(multivariate_normal.pdf)
profile.add_function(multivariate_normal._logpdf)
profile.add_function(MVN.__call__)
profile.enable_by_count()
# mean_shift(X, maxiter=1000)
MVN(delta)
multivariate_normal.pdf(delta, mean=mu, cov=cov)
profile.print_stats()
