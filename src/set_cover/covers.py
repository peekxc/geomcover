from math import comb
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, csc_matrix, find, coo_matrix, csc_array
from scipy.spatial.distance import pdist, cdist, squareform
from combin import inverse_choose

## Predicates to simplify type-checking
def is_distance_matrix(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a distance matrix, i.e. is square, symmetric, and that the diagonal is all 0. '''
	x = np.array(x, copy=False)
	is_square = x.ndim == 2	and (x.shape[0] == x.shape[1])
	return(False if not(is_square) else np.all(np.diag(x) == 0))

def is_pairwise_distances(x: ArrayLike) -> bool:
	''' Checks whether 'x' is a 1-d array of pairwise distances '''
	x = np.array(x, copy=False) # don't use asanyarray here
	if x.ndim > 1: return(False)
	n = inverse_choose(len(x), 2)
	return(x.ndim == 1 and len(x) == comb(n, 2))

def is_point_cloud(x: ArrayLike) -> bool: 
	''' Checks whether 'x' is a 2-d array of points '''
	return(isinstance(x, np.ndarray) and x.ndim == 2)

## Classical MDS 
def cmds(G: ArrayLike, d: int = 2, coords: bool = True):
	''' Projects 'G' onto a d-dimensional embedding via Classical (Torgerson's) Multi-Dimensional Scaling (CMDS)

	CMDS is a linear dimensionality reduction algorithm that projects a double-centered symmetric inner product (gram) matrix 'G' to a 
	lower dimensional space whose coordinates minimize the reconstruction error of centered scalar products, or 'strain'.

	Parameters: 
		G = set of pairwise inner products or a squared Euclidean distance matrix. 
		d = dimension of the embedding to produce.
		center = whether to center the data prior to computing eigenvectors
		coords = whether to return the embedding (default = True), or just return the eigenvectors
	
	Returns:
		if coords = True, returns the projection of 'X' onto the largest 'd' eigenvectors of X's covariance matrix. Otherwise, 
		the eigenvalues and eigenvectors can be returned as-is. 
	''' 
	is_pd = is_pairwise_distances(G)
	is_dm = is_distance_matrix(G)
	assert is_pd or is_dm, "Input 'D' should set of pairwise distances or distance matrix"
	G = squareform(G) if is_pd else G
	n = G.shape[0]
	G_center = G.mean(axis=0)
	G = -0.50 * (G  - G_center - G_center.reshape((n,1)) + G_center.mean())
	evals, evecs = np.linalg.eigh(G)
	evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]

	# Compute the coordinates using positive-eigenvalued components only     
	if coords:               
		w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
		Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
		return(Y)
	else: 
		w = np.where(evals > 0)[0]
		ni = np.setdiff1d(np.arange(d), w)
		evecs[:,ni] = 1.0
		evals[ni] = 0.0
		return(evals, evecs)

def pca(X: ArrayLike, d: int = 2, center: bool = False, coords: bool = True) -> ArrayLike:
	''' 
	Projects 'X' onto a d-dimensional embedding via Principal Component Analysis (PCA)

	PCA is a linear dimensionality reduction algorithm that projects a point set 'X' onto a lower dimensional space 
	using an orthogonal projector built from the eigenvalue decomposition of its covariance matrix. 

	PCA is dual to CMDS in the sense that the d-dimensional embedding produced by CMDS on the gram matrix  
	of squared Euclidean distances from 'X' satisfies the same reconstruction as the d-dimensional projection of 'X' with PCA. 

	Parameters: 
		X = (n x D) point cloud / design matrix of n points in D dimensions. 
		d = dimension of the embedding to produce.
		center = whether to center the data prior to computing eigenvectors
		coords = whether to return the embedding (default), or just return the eigenvectors
	
	Returns:
		if coords = True (default), returns the projection of 'X' onto the largest 'd' eigenvectors of X's covariance matrix. 
		Otherwise, the eigenvalues and eigenvectors can be returned as-is. 
	'''
	X = np.atleast_2d(X)
	assert is_point_cloud(X), "Input should be a point cloud, not a distance matrix."
	if center: 
		X -= X.mean(axis = 0)
	evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
	idx = np.argsort(evals)[::-1] # descending order to pick the largest components first 
	if coords:
		return(np.dot(X, evecs[:,idx[range(d)]]))
	else: 
		return(evals[idx[range(d)]], evecs[:,idx[range(d)]])

def tangent_neighbor_graph(X: ArrayLike, d: int, r: float, ind = None):
	''' 
	Constructs an r-neighborhood graph on the point cloud 'X' at the given indices 'ind', and then computes an orthogonal basis 
	which approximates the d-dimensional tangent space around each of those points. 

	Parameters: 
		X = (n x d) point cloud data in Euclidean space, or and (n x n) sparse adjacency matrix yielding a weighted neighborhood graph
		d = local dimension where the metric is approximately Euclidean
		r = radius around each point determining the neighborhood from which to compute the tangent vector
		ind = indices to approximate the neighborhoods at. If not specified, will use every point. 

	Returns: 
		G = the neighborhood graph, given as an (n x len(ind)) incidence matrix
		weights = len(ind)-length array 
	'''
	ind = np.array(range(X.shape[0])) if ind is None else ind
	m = len(ind)
	r,c,v = find(cdist(X, X[ind,:]) <= r*2)
	weights, tangents = np.zeros(m), [None]*m
	for i, x in enumerate(X[ind,:]): 
		nn_idx = c[r == i] #np.append(np.flatnonzero(G[i,:].A), i)
		if len(nn_idx) < 2: 
			# raise ValueError("Singularity at point {i}: neighborhood too small to compute tangent")
			weights[i] = np.inf 
			tangents[i] = np.zeros(shape=(X.shape[1], d))
			continue 
		
		## Get tangent space estimates at centered points
		centered_pts = X[nn_idx,:]-x
		_, T_y = pca(centered_pts, d=d, coords=False)
		tangents[i] = T_y # ambient x local

		## Project all points onto tangent plane, then measure distance between projected points and original
		proj_coords = np.dot(centered_pts, T_y) # project points onto d-tangent plane
		proj_points = np.array([np.sum(p*T_y, axis=1) for p in proj_coords]) # orthogonal projection in D dimensions
		weights[i] = np.sum([np.sqrt(np.sum(diff**2)) for diff in (centered_pts - proj_points)]) # np.linalg.norm(centered_pts - proj_points)
	
	#assert np.all(G.A == G.A.T)
	G = coo_matrix((v, (r,c)), shape=(X.shape[0], len(ind)), dtype=bool)
	return(G.tocsc(), weights, tangents)
	#return(weights, tangents)

def tangent_bundle(G, X: np.ndarray) -> dict:
	'''Estimates the tangent bundle of a graph 'G' whose vertices in Euclidean space 'X' via local PCA.'''
	pass 
	# ## Get tangent space estimates at centered points
	# centered_pts = X[nn_idx,:]-x
	# _, T_y = pca(centered_pts, d=d, coords=False)
	# tangents[i] = T_y # ambient x local
	# if len(nn_idx) < 2: 
	# # raise ValueError("Singularity at point {i}: neighborhood too small to compute tangent")
	# weights[i] = np.inf 
	# tangents[i] = np.zeros(shape=(X.shape[1], d))
	# continue 


def valid_cover(A, ind: np.ndarray = None) -> bool:
	"""Determines whether certain subsets of a set of subsets forms a covers every row."""
	import sortednp
	n, J = A.shape
	A = csc_array(A).astype(bool) if not hasattr(A, "indices") else A
	A.eliminate_zeros()
	A.sort_indices()
	subset_splits = np.split(A.indices, A.indptr)[1:-1]
	assert len(subset_splits) == J, "Splitting of cover array failed. Are there empty columns?"
	if ind is not None:
		ind = np.array(ind).astype(int) 
		subset_splits = [subset_splits[i] for i in ind]
	covered_ind = sortednp.kway_merge(*subset_splits, assume_sorted=True, duplicates=4)
	return len(covered_ind) == n