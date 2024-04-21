import math
import numpy as np
from numpy.typing import ArrayLike
# from tallem.dimred import neighborhood_graph, neighborhood_list, pca
from scipy.sparse import csr_matrix, csc_matrix, find, coo_matrix
from scipy.spatial.distance import pdist, cdist


## Utility 
def inverse_choose(x: int, k: int):
	assert k >= 1, "k must be >= 1" 
	if k == 1: return(x)
	if k == 2:
		rng = np.array(list(range(int(np.floor(np.sqrt(2*x))), int(np.ceil(np.sqrt(2*x)+2) + 1))))
		final_n = rng[np.nonzero(np.array([math.comb(n, 2) for n in rng]) == x)[0].item()]
	else:
		# From: https://math.stackexchange.com/questions/103377/how-to-reverse-the-n-choose-k-formula
		if x < 10**7:
			lb = (math.factorial(k)*x)**(1/k)
			potential_n = np.array(list(range(int(np.floor(lb)), int(np.ceil(lb+k)+1))))
			idx = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[idx]
		else:
			lb = np.floor((4**k)/(2*k + 1))
			C, n = math.factorial(k)*x, 1
			while n**k < C: n = n*2
			m = (np.nonzero( np.array(list(range(1, n+1)))**k >= C )[0])[0].item()
			potential_n = np.array(list(range(int(np.max([m, 2*k])), int(m+k+1))))
			ind = np.nonzero(np.array([math.comb(n, k) for n in potential_n]) == x)[0].item()
			final_n = potential_n[ind]
	return(final_n)

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
	return(x.ndim == 1 and n == int(n))

def is_point_cloud(x: ArrayLike) -> bool: 
	''' Checks whether 'x' is a 2-d array of points '''
	return(isinstance(x, np.ndarray) and x.ndim == 2)

## Classical MDS 
def cmds(a: ArrayLike, d: int = 2, coords: bool = True):
	''' Computes classical MDS (cmds) '''
	if is_pairwise_distances(a):
		D = as_dist_matrix(a)
	elif not(is_distance_matrix(a)) and is_point_cloud(a):
		D = cdist(a, a)
	else:
		D = a
	assert(is_distance_matrix(D))
	n = D.shape[0]
	D_center = D.mean(axis=0)
	Dc = -0.50 * (D  - D_center - D_center.reshape((n,1)) + D_center.mean())
	evals, evecs = np.linalg.eigh(Dc)
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

## Plain PCA
def pca(x: ArrayLike, d: int = 2, center: bool = False, coords: bool = True) -> ArrayLike:
	''' 
	Uses PCA to produce a d-dimensional embedding of 'x', or an eiegnvalue decomposition of its covariance matrix. 
	
	Parameters: 
		x := point cloud, distance matrix, or set of pairwise distances
		d := dimension of the embedding 
		center := whether to center the data prior to computing eigenvectors
		coords := whether to return the embedding (default), or just return the eigenvectors
	
	Returns: 
		if coords = False, returns the projection of 'x' onto the largest 'd' eiegenvectors of x's covariance matrix. Otherwise, 
		the eigenvalues and eigenvectors can be returned as-is. 
	'''
	if is_pairwise_distances(x) or is_distance_matrix(x):
		return(cmds(x, d))
	assert is_point_cloud(x), "Input should be a point cloud, not a distance matrix."
	if center: x -= x.mean(axis = 0)
	evals, evecs = np.linalg.eigh(np.cov(x, rowvar=False))
	idx = np.argsort(evals)[::-1] # descending order to pick the largest components first 
	if coords:
		return(np.dot(x, evecs[:,idx[range(d)]]))
	else: 
		return(evals[idx[range(d)]], evecs[:,idx[range(d)]])

def tangent_neighbor_graph(X: ArrayLike, d: int, r: float, ind = None):
	''' 
	Constructs an r-neighborhood graph on the point cloud 'X' at the given indices 'ind', and then computes an orthogonal basis 
	which approximates the d-dimensional tangent space around each of those points. 

	Parameters: 
		X := (n x d) point cloud data in Euclidean space, or and (n x n) sparse adjacency matrix yielding a weighted neighborhood graph
		d := local dimension where the metric is approximately Euclidean
		r := radius around each point determining the neighborhood from which to compute the tangent vector
		ind := indices to approximate the neighborhoods at. If not specified, will use every point. 

	Returns: 
		G := (n x len(ind)) csc_matrix giving the incidence relations of the neighborhood graph
		weights := len(ind)-length array 
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
