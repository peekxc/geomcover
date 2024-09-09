"""Provides standard linear algebra algorithms."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import squareform
from typing import Union


## Classical MDS
## See also: https://math.stackexchange.com/questions/3704727/derivative-of-double-centered-euclidean-distance-matrix
def cmds(D: ArrayLike, d: int = 2, coords: bool = True) -> Union[np.ndarray, tuple]:
	"""Constructs coordinates from squared distances using Classical Multi-Dimensional Scaling.

	CMDS is a coordinatization algorithm that generates `d`-dimensional coordinates from a Euclidean \
	distance matrix `D`. Algorithmically, `D` is converted into a doubly-centered Gram matrix `G` \
	whose eigen decomposition is used to produce coordinates minimizing a notion of 'strain'.

	Parameters: 
		D: Squared Euclidean distance matrix, or set of (squared) pairwise distances. 
		d: dimension of the embedding to produce.
		coords: whether to return the embedding (default = True), or just return the eigenvectors
	
	Returns:
		if `coords = True`, the coordinates from the largest `d` eigenvectors of `G`'s eigendecomposition. Otherwise, \
		the eigenvalues and eigenvectors are returned. See the Examples section for more details. 
	
	Notes:
		PCA is dual to CMDS in the sense that the embedding produced by CMDS on the Euclidean distance matrix from `X` \
		satisfies the same reconstruction loss as with PCA. In particular, when `X` comes from Euclidean space, \
		the output of pca(...) will match the output of cmds(...) exactly up to rotation and translation. 
	
	See Also:
		- [CMDS](https://en.wikipedia.org/wiki/Multidimensional_scaling)
		- [Euclidean distance matrix](https://en.wikipedia.org/wiki/Euclidean_distance_matrix)
	
	Examples:
		```{python}
		import numpy as np 
		from geomcover.linalg import pca, cmds

		## Start with a random set of points in R^3 + its distance matrix
		X = np.random.uniform(size=(50,3))
		D = np.linalg.norm(X - X[:,np.newaxis], axis=2)
		
		## Note that CMDS takes as input *squared* distances
		Y_pca = pca(X, d=2)
		Y_mds = cmds(D**2, d=2)

		## Get distance matrices for both embeddings
		Y_pca_D = np.linalg.norm(Y_pca - Y_pca[:,np.newaxis], axis=2)
		Y_mds_D = np.linalg.norm(Y_mds - Y_mds[:,np.newaxis], axis=2)
		
		## Up to rotation and translation, the coordinates are identical
		all_close = np.allclose(Y_pca_D, Y_mds_D)
		print(f"PCA and MDS coord. distances identical? {all_close}")
		```
	"""
	D = np.asarray(D)
	D = squareform(D) if D.ndim == 1 else D
	n = D.shape[0]
	D_center = D.mean(axis=0)
	G = -0.50 * (D - D_center - D_center.reshape((n, 1)) + D_center.mean())
	evals, evecs = np.linalg.eigh(G)
	evals, evecs = evals[(n - d) : n], evecs[:, (n - d) : n]

	# Compute the coordinates using positive-eigenvalued components only
	if coords:
		w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
		Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
		return Y
	else:
		w = np.where(evals > 0)[0]
		ni = np.setdiff1d(np.arange(d), w)
		evecs[:, ni] = 1.0
		evals[ni] = 0.0
		return (evals, evecs)


def pca(X: ArrayLike, d: int = 2, center: bool = True, coords: bool = True) -> Union[np.ndarray, tuple]:
	"""Projects `X` onto a `d`-dimensional linear subspace via Principal Component Analysis.

	PCA is a linear dimensionality reduction algorithm that projects a point set `X` onto a lower dimensional space \
	using an orthogonal projector built from the eigenvalue decomposition of its covariance matrix. 

	Parameters: 
		X: (n x D) point cloud / design matrix of n points in D dimensions. 
		d: dimension of the embedding to produce.
		center: whether to center the data prior to computing eigenvectors. Defaults to True. 
		coords: whether to return the embedding or the eigenvectors. Defaults to the embedding. 
	
	Returns:
		if coords = True (default), returns the projection of `X` onto the largest `d` eigenvectors of `X`s covariance matrix. \
		Otherwise, the eigenvalues and eigenvectors can be returned as-is. 

	Notes:
		PCA is dual to CMDS in the sense that the embedding produced by CMDS on the Euclidean distance matrix from `X` \
		satisfies the same reconstruction loss as with PCA. In particular, when `X` comes from Euclidean space, \
		the output of pca(...) will match the output of cmds(...) exactly up to rotation and translation. 
	
	Examples:
		```{python}
		import numpy as np 
		from geomcover.linalg import pca, cmds

		## Start with a random set of points in R^3 + its distance matrix
		X = np.random.uniform(size=(50,3))
		D = np.linalg.norm(X - X[:,np.newaxis], axis=2)
		
		## Note that CMDS takes as input *squared* distances
		Y_pca = pca(X, d=2)
		Y_mds = cmds(D**2, d=2)

		## Get distance matrices for both embeddings
		Y_pca_D = np.linalg.norm(Y_pca - Y_pca[:,np.newaxis], axis=2)
		Y_mds_D = np.linalg.norm(Y_mds - Y_mds[:,np.newaxis], axis=2)
		
		## Up to rotation and translation, the coordinates are identical
		all_close = np.allclose(Y_pca_D, Y_mds_D)
		print(f"PCA and MDS coord. distances identical? {all_close}")
		```
	"""
	X: np.ndarray = np.atleast_2d(X)
	if center:
		X -= X.mean(axis=0)
	evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
	idx = np.argsort(evals)[::-1]  # descending order to pick the largest components first
	if coords:
		return np.dot(X, evecs[:, idx[:d]])
	else:
		return (evals[idx[:d]], evecs[:, idx[:d]])
