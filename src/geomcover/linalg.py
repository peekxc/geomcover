"""Provides standard linear algebra algorithms."""

import numpy as np 
from numpy.typing import ArrayLike
from scipy.spatial.distance import squareform

## Classical MDS 
def cmds(G: ArrayLike, d: int = 2, coords: bool = True) -> np.ndarray:
	"""Projects `G` onto a `d`-dimensional linear subspace via Classical Multi-Dimensional Scaling.

	CMDS is a linear dimensionality reduction algorithm that projects a double-centered symmetric \
	inner product (gram) matrix 'G' to a lower dimensional space whose coordinates minimize the  \
	reconstruction error of centered scalar products, or 'strain'.

	CMDS is dual to PCA in the sense that the covariance-derived projection produced by PCA minimizes the 
	same inner-product-derived 'strain' objective minimized by CMDS. 

	Parameters: 
		G: set of pairwise inner products or a squared Euclidean distance matrix. 
		d: dimension of the embedding to produce.
		coords: whether to return the embedding (default = True), or just return the eigenvectors
	
	Returns:
		if coords = True, returns the projection of 'X' onto the largest 'd' eigenvectors of X's covariance matrix. Otherwise, 
		the eigenvalues and eigenvectors can be returned as-is. 
	""" 
	G = np.asarray(G)
	G = squareform(G) if G.ndim == 1 else G
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

def pca(X: ArrayLike, d: int = 2, center: bool = False, coords: bool = True) -> np.ndarray:
	"""Projects `X` onto a d-dimensional linear subspace via Principal Component Analysis.

	PCA is a linear dimensionality reduction algorithm that projects a point set 'X' onto a lower dimensional space 
	using an orthogonal projector built from the eigenvalue decomposition of its covariance matrix. 

	PCA is dual to CMDS in the sense that the d-dimensional embedding produced by CMDS on the gram matrix of squared \
	Euclidean distances from 'X' satisfies the same reconstruction as the d-dimensional projection of 'X' with PCA. 

	Parameters: 
		X: (n x D) point cloud / design matrix of n points in D dimensions. 
		d: dimension of the embedding to produce.
		center: whether to center the data prior to computing eigenvectors.
		coords: whether to return the embedding (default), or just return the eigenvectors.
	
	Returns:
		if coords = True (default), returns the projection of 'X' onto the largest 'd' eigenvectors of X's covariance matrix. 
		Otherwise, the eigenvalues and eigenvectors can be returned as-is. 
	"""
	X = np.atleast_2d(X)
	if center: 
		X -= X.mean(axis = 0)
	evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
	idx = np.argsort(evals)[::-1] # descending order to pick the largest components first 
	if coords:
		return(np.dot(X, evecs[:,idx[range(d)]]))
	else: 
		return(evals[idx[range(d)]], evecs[:,idx[range(d)]])
