"""Geometry algorithms for manifolds."""

from collections import namedtuple
from itertools import combinations
from math import comb
from typing import Callable, Union, Optional

import numpy as np
from combin import inverse_choose
from numpy.typing import ArrayLike
from scipy.sparse import coo_array, coo_matrix, csr_array, find, sparray
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from .io import _ask_package_install, to_canonical
from .linalg import pca

## Simple type
TangentPair = namedtuple("TangentPair", field_names=("base_point", "tangent_vec"))


## Predicates to simplify type-checking
def is_distance_matrix(x: ArrayLike) -> bool:
	"""Checks whether `x` is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0."""
	x = np.array(x, copy=False)
	is_square = x.ndim == 2 and (x.shape[0] == x.shape[1])
	return False if not (is_square) else bool(np.all(np.diag(x) == 0))


def is_pairwise_distances(x: ArrayLike) -> bool:
	"""Checks whether `x` is a 1-d array of pairwise distances."""
	x = np.array(x, copy=False)  # don't use asanyarray here
	if x.ndim > 1:
		return False
	n = inverse_choose(len(x), 2)
	return x.ndim == 1 and len(x) == comb(n, 2)


def is_point_cloud(x: ArrayLike) -> bool:
	"""Checks whether `x` is a 2-d array of points."""
	return isinstance(x, np.ndarray) and x.ndim == 2


# def nerve():
# 	"""Computes the simplicial nerve of a given cover"""

# def neighborhood_graph(X: np.ndarray, r: float, ind = None):
# 	"""Constructs an 'r'-neighborhood graph on the point cloud 'X' at the given indices 'ind'

# 	Returns:
# 		G = compressed (n x i) adjacency list, where n = |X| and i = |ind|, given as a CSR sparse matrix
# 		r = radius of the ball to thicken each point in X with
# 	"""
# 	ind = np.array(range(X.shape[0])) if ind is None else ind
# 	m = len(ind)
# 	r,c,v = find(cdist(X, X[ind,:]) <= r*2)
# 	G = coo_matrix((v, (r,c)), shape=(X.shape[0], m), dtype=bool)
# 	return G.tocsc()


def tangent_bundle(M: sparray, X: np.ndarray, d: int = 2, centers: Optional[np.ndarray] = None) -> dict:
	"""Estimates the tangent bundle of a range space (`X`,`M`) via local PCA.

	This function estimates the `d`-dimensional tangent spaces of neighborhoods in `X` given by ranges in `M`.
	This may be interpreted as evaluating the logarithm between the `centers` and points in the local neighborhood
	in the direction given by the principle directions.

	Parameters:
		M: Sparse matrix whose columns represent subsets of `X`.
		X: coordinates of the range space.
		d: dimension of the tangent space.
		centers: points to center the tangent space estimates. If `None`, each neighborhoods is centered around its average.

	Returns:
		list of *tangent pairs*, i.e. base points paired with tangent vector.
	"""
	M = to_canonical(M, "csr")
	if centers is not None:
		centers = np.atleast_2d(centers)
		assert centers.shape[1] == X.shape[1], "Centers must have same dimension as 'X'"
		assert len(centers) == M.shape[1], "Number of centers doesn't match number of neighborhoods in 'M'"
	D = X.shape[1]
	m = M.shape[1]
	tangents = [None] * m
	for j, ind in enumerate(np.split(M.indices, M.indptr)[1:-1]):
		## First compute the base point
		center = centers[j] if centers is not None else X[ind, :].mean(axis=0)
		if len(ind) <= 1:
			# raise ValueError("Singularity at point {i}: neighborhood too small to compute tangent")
			tangents[j] = TangentPair(center, np.eye(D, d))
			continue

		## Get tangent space estimates at centered points
		centered_pts = X[ind, :] - center
		_, T_y = pca(centered_pts, d=d, coords=False)
		tangents[j] = TangentPair(center, T_y)  # ambient x local, columns represent unit vectors
	return tangents


def bundle_weights(M: sparray, TM: list, method: str, reduce: Union[str, Callable]) -> np.ndarray:
	"""Computes geometrically informative statistics about a given tangent bundle.

	This function computes a geometrically-derived statistic about a given tangent space `TM` using its neighborhood information. Such \
	measures can at times be useful for constructing nerve complexes, removing outliers, detecting locally smooth areas, etc. 

	The methods supported include 'distance', 'cosine', and 'angle', which do the following:
	
		* 'distance': Measures the distance from each neighborhood point to its projection onto the tangent space using the Euclidean norm.
		* 'cosine': Measures the distance from each neighborhood tangent vector to a fixed tangent vector using the cosine distance. 
		* 'angle': Measures the distance from each neighborhood tangent vector to a fixed tangent vector using the stiefel canonical metric. 

	Parameters:
		M: Sparse matrix whose columns represent subsets of `X`.
		TM: Tangent bundle, given as a list of _(base point, tangent vector)_ pairs
		method: geometric quantity to compute, one of `{'distance', 'cosine', 'angle'}`. Defaults to 'cosine'. 
		reduce: aggregation function to compute the final statistic. Defaults to the average (see details).

	Returns:
		the aggregate statistic for each tangent space, given as an array of weights.
	"""
	A = M.tocoo()
	assert method in {"distance", "cosine", "angle"}, f"Invalid method '{method!s}' supplied"
	stat_f = getattr(np, reduce) if isinstance(reduce, str) else reduce
	assert isinstance(
		stat_f, Callable
	), "Reduce function should be the name of a numpy aggregate function or a Callable itself."

	if method == "cosine":
		## The cosine distance is taken as the minimum to each tangent vector and its opposite
		base_points = np.array([p for p, v in TM])  # n x D
		tangent_vec = np.array([v.T.flatten() for p, v in TM])  # n x D x d
		TV, DN = tangent_vec, np.array([[1], [-1]])
		cosine_dist_sgn = lambda j: np.min(cdist(TV[[j, j]] * DN, TV[A.row[A.col == j]], "cosine"), axis=0)  # noqa: E731
		cosine_dist = [cosine_dist_sgn(j) for j in range(M.shape[1])]
		weights = np.array([stat_f(cd) for cd in cosine_dist])
		return weights
	elif method == "distance":
		base_points = np.array([p for p, v in TM])  # n x D
		tangent_vec = np.array([v.T.flatten() for p, v in TM])  # n x D x d
		weights = np.zeros(len(TM), dtype=np.float32)
		for j, (pt, T_y) in enumerate(TM):
			neighbor_ind = A.row[A.col == j]
			neighbor_coords = base_points[neighbor_ind]

			## Collect dist to each tangent line
			proj_dist = np.zeros(shape=(len(neighbor_ind), T_y.shape[1]))
			for ii, tangent_v in enumerate(T_y.T):
				tangent_inner_prod = (neighbor_coords - pt).dot(tangent_v)
				proj_coords = pt + tangent_v * tangent_inner_prod[:, np.newaxis]
				proj_dist[:, ii] = np.linalg.norm(proj_coords - neighbor_coords, axis=1)
			weights[j] = stat_f(proj_dist)
		return weights
	else:
		_ask_package_install("geomstats")  ## ensures package is installed, asks if not
		from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric  # noqa: PLC0415

		manifold = Stiefel(*TM[0].tangent_vec.shape)  # 1-frames in R^2
		metric = StiefelCanonicalMetric(manifold)
		weights = np.zeros(len(TM), dtype=np.float32)
		for j, (pt, T_y) in enumerate(TM):
			neighbor_ind = A.row[A.col == j]
			tangent_prods = np.abs([metric.inner_product(T_y, TM[i].tangent_vec, base_point=T_y) for i in neighbor_ind])
			# print(f"{j} => {tangent_prods}")
			## All entries should be in [-k, k], where k = 0.5 * p
			# (0.5 * manifold.p)
			tangent_prods = (0.5 * manifold.p) - tangent_prods
			weights[j] = stat_f(tangent_prods)
		return weights


def neighbor_graph_ball(
	X: ArrayLike, radius: float, batch: int = 15, metric: str = "euclidean", weighted: bool = False, **kwargs
):
	"""Constructs a neighborhood graph by via the nerve of the union of balls centered at `X`."""
	from array import array

	n = len(X)
	R, C = array("I"), array("I")
	V = array("f") if weighted else []
	threshold = 2.0 * radius
	dtype = np.float32 if weighted else bool
	for ind in np.array_split(range(n), n // batch):
		D = cdist(X, X[ind, :], metric=metric, **kwargs)
		r, c, v = find(np.where(D <= threshold, D, 0.0))
		R.extend(r)
		C.extend(ind[c])
		V.extend(v.astype(dtype))
	G = coo_matrix((V, (R, C)), shape=(n, n), dtype=dtype)
	return to_canonical(G, form="csc", diag=True, symmetrize=False)


def neighbor_graph_knn(
	X: ArrayLike, k: int, batch: int = 15, metric: str = "euclidean", weighted: bool = False, diag: bool = False, **kwargs
):
	"""Empty."""
	from array import array

	n = len(X)
	R, C = array("I"), array("I")
	V = array("f") if weighted else array("B")
	for ind in np.array_split(range(n), n // batch):
		D = cdist(X[ind, :], X, metric=metric, **kwargs)
		r = D.argpartition(kth=(k + 1), axis=1)[:, : (k + 1)].flatten()
		c = np.repeat(ind, k + 1)
		R.extend(r)
		C.extend(c)
		V.extend(D[c - c[0], r] if weighted else np.ones(len(r), dtype=bool))
	R, C = np.array(R), np.array(C)
	dtype = np.float32 if weighted else bool
	G = coo_matrix((V, (R, C)), shape=(n, n), dtype=dtype)
	return to_canonical(G, form="csc", diag=True, symmetrize=False)


def neighbor_graph_del(X: ArrayLike, weighted: bool = False, **kwargs):
	"""Empty."""
	from array import array

	n = len(X)
	dt = Delaunay(X)
	R, C = array("I"), array("I")
	for I_ind, J_ind in combinations(dt.simplices.T, 2):
		R.extend(I_ind)
		C.extend(J_ind)
	V = np.ones(len(R)) if not weighted else np.linalg.norm(X[R] - X[C], axis=1)
	dtype = np.bool if not weighted else np.float32
	G = coo_array((V, (R, C)), shape=(n, n), dtype=dtype)
	return to_canonical(G, form="csc", diag=True, symmetrize=False)


def tangent_neighbor_graph(X: ArrayLike, d: int, r: float, ind=None):
	"""Constructs an r-neighborhood graph on the point cloud 'X' at the given indices 'ind', and then computes an orthogonal basis
	which approximates the d-dimensional tangent space around each of those points.

	Parameters:
		X = (n x d) point cloud data in Euclidean space, or and (n x n) sparse adjacency matrix yielding a weighted neighborhood graph
		d = local dimension where the metric is approximately Euclidean
		r = radius around each point determining the neighborhood from which to compute the tangent vector
		ind = indices to approximate the neighborhoods at. If not specified, will use every point.

	Returns:
		neighborhood graph and weights.
	"""
	ind = np.array(range(X.shape[0])) if ind is None else ind
	m = len(ind)
	r, c, v = find(cdist(X, X[ind, :]) <= r * 2)
	weights, tangents = np.zeros(m), [None] * m
	for i, x in enumerate(X[ind, :]):
		nn_idx = c[r == i]  # np.append(np.flatnonzero(G[i,:].A), i)
		if len(nn_idx) < 2:
			# raise ValueError("Singularity at point {i}: neighborhood too small to compute tangent")
			weights[i] = np.inf
			tangents[i] = np.eye(X.shape[1], d)
			continue

		## Get tangent space estimates at centered points
		centered_pts = X[nn_idx, :] - x
		_, T_y = pca(centered_pts, d=d, coords=False)
		tangents[i] = T_y  # ambient x local

		## Project all points onto tangent plane, then measure distance between projected points and original
		proj_coords = np.dot(centered_pts, T_y)  # project points onto d-tangent plane
		proj_points = np.array([np.sum(p * T_y, axis=1) for p in proj_coords])  # orthogonal projection in D dimensions
		weights[i] = np.sum(
			[np.sqrt(np.sum(diff**2)) for diff in (centered_pts - proj_points)]
		)  # np.linalg.norm(centered_pts - proj_points)

	# assert np.all(G.A == G.A.T)
	G = coo_matrix((v, (r, c)), shape=(X.shape[0], len(ind)), dtype=bool)
	return (G.tocsc(), weights, tangents)
	# return(weights, tangents)
