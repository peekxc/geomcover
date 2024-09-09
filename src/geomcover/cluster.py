import math
from functools import partial
from numbers import Integral
from typing import Callable, Collection, Generator, Iterable, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.stats import Covariance, gaussian_kde, multivariate_normal


class MultivariateNormalZeroMean:
	## About 130% faster than scipy multivariate_normal.pdf with fixed Covariance
	def __init__(self, cov: Covariance) -> None:
		# assert "CovViaPrecision" in str(type(cov))
		self.cov = cov
		self._LOG_2PI = np.log(2 * np.pi)
		self.rank = cov.rank[..., np.newaxis]

	## Optimizations attempted:
	# - Surprisingly, np.dot(..., out=...) with pre-allocated out was slower than @ (used internally with whiten)!
	# - whiten and sum and exp are the three bottlenecks
	# - Inplace can help relative to the big composition of temporaries below
	# - eigen or precision covariance is preferred
	# - Maybe pythran could help
	# Slower options:
	# maha = np.sum(np.square(self.cov.whiten(dev)), axis=-1)
	# np.dot(x, self.cov._chol_P, out=self._whiten_out)
	# return np.exp(-0.5 * (np.sum(np.square(_whiten_out), axis=-1) + self.rank * self._LOG_2PI + log_det_cov))
	def __call__(self, x: np.ndarray) -> np.ndarray:
		log_det_cov = self.cov.log_pdet[..., np.newaxis] if x.ndim > 1 else self.cov.log_pdet
		_whiten_out = self.cov.whiten(x)
		np.square(_whiten_out, out=_whiten_out)
		_maha = _whiten_out.sum(axis=-1)
		_maha += self.rank * self._LOG_2PI + log_det_cov
		_maha *= -0.5
		np.exp(_maha, out=_maha)
		return _maha


def assign_clusters(shift_points: np.ndarray, atol: float = 1.1920929e-05) -> np.ndarray:
	"""Aggregates the converged points into clusters."""
	rng = np.random.default_rng()
	cl: np.ndarray = np.nan * np.ones(len(shift_points))
	ind = np.array([0], dtype=np.int64)
	cid = 0
	while np.any(np.isnan(cl)):
		sp = shift_points[rng.choice(ind, size=1).item()]
		converge_mask = np.linalg.norm(sp - shift_points, axis=1) <= atol
		converge_mask &= np.isnan(cl)
		cl[converge_mask] = cid
		cid += 1
		ind = np.flatnonzero(np.isnan(cl))
	return cl


## Same as above, but vectorized to do multiple points at a time
def batch_shift_kernel(P: np.ndarray, points: np.ndarray, kernel: Callable) -> np.ndarray:
	"""Shift operator: batch-shifts `P` with respect to their `kernel`-distance to `points`."""
	delta = P[:, np.newaxis, :] - points[np.newaxis, :, :]
	P_weights = np.atleast_2d(kernel(delta))
	denom = np.sum(P_weights, axis=1)
	return (P_weights @ points) / denom[:, np.newaxis]


## TODO: gaussian_kde actually gives denom, but not P_weights. Unclear if we can get by with it.
## It is potentially 6x+ faster.
# def batch_shift_kde(P: np.ndarray, kde: Callable) -> np.ndarray:
# 	"""Shift operator: batch-shifts `P` with respect to their `kernel`-distance to `points`."""
# 	# assert hasattr(kde, "evaluate"), "Must be a valid KDE object"
# 	P_weights = kde.evaluate(P)
# 	denom = np.sum(P_weights, axis=1)
# 	return (P_weights @ points) / denom[:, np.newaxis]


## Equivalent to more itertools
def chunked(iterable: Union[Sequence, np.ndarray], n: int) -> Generator:
	for start in range(0, len(iterable), n):
		yield iterable[start : start + n]


## Heavily modified from: https://github.com/mattnedrich/MeanShift_py/tree/master
## Optimizations attempted:
# cov = Covariance.from_eigendecomposition(np.linalg.eigh(KDE.covariance))
# cov = Covariance.from_cholesky(KDE.cho_cov)
# kernel = partial(multivariate_normal.pdf, mean=mu, cov=cov)
def mean_shift(
	P: np.ndarray,
	Q: Optional[np.ndarray] = None,
	kernel: Union[str, Callable] = "gaussian",
	bandwidth: Union[str, float, Callable] = "scott",
	batch: int = 25,
	maxiter: Optional[float] = 300,
	atol: float = 1.1920929e-06,
	blur: bool = False,
	callback: Optional[Callable] = None,
) -> np.ndarray:
	"""Mean shift algorithm for clustering or smoothing points via kernel density estimation.

	This functions repeatedly mean shifts points `P` with respect to reference points `Q`, returning the shifted points. \
	If `Q` is not supplied the shift points are shifted w.r.t `P`, or the shift points themselves if `blur = True`, until \
	either `maxiter` iterations is reached the distance shifted is less than `atol` (elementwise), whichever comes first. 

	Parameters:
		P: points to shift, given as a 2d np.array. 
		Q: reference points to shift `P` with respect to. Defaults to `P` itself. 
		kernel: kernel function to use. Only "gaussian" is supported for now. 
		bandwidth: smoothing parameter for the kernel.
		batch: number of points to apply the shift to at once. See details. 
		maxiter: maximum number of times to apply a the shift to a point. Can be `None` or `np.inf` to 
		atol: absolute tolerance with which to consider a point as converged between iterations.
		blur: whether to apply the blurring mean shift operation. See details. 
		callback: callable to execute after each iteration.

	Returns:
		ndarray of shift points with the same dimensions as `P`. 
	"""
	assert not (blur) or Q is None, "`Q` cannot be supplied when `blur = True`."
	assert kernel == "gaussian", "Only Gaussian kernels are supported currently"

	## Determine shift points and reference points
	SP = np.asarray(P) if not isinstance(P, np.ndarray) else P
	RP = SP if Q is None else np.asarray(Q)

	## Constants
	N, d = SP.shape
	maxiter = np.inf if maxiter is None else maxiter

	## Setup covariance for fast calculations
	## NOTE: this doesn't actually evaluate the KDE, just uses initializer for validation + covariance + bandwidth resolution
	KDE = gaussian_kde(RP.T, bw_method=bandwidth)
	cov = Covariance.from_precision(KDE.inv_cov, KDE.covariance)
	kernel = MultivariateNormalZeroMean(cov)
	# kernel = lambda x: kernel_pdf(x) / N ## To match the scale of multivariate_normal.pdf
	# kernel = partial(multivariate_normal.pdf, mean=np.zeros(d), cov=cov)

	## Setup initial parameters
	it = 0
	shift_points = SP.copy()
	still_shifting = np.ones(N, dtype=bool)
	while np.any(still_shifting) and it < maxiter:
		it += 1
		for ind in chunked(np.flatnonzero(still_shifting), batch):
			p_new = batch_shift_kernel(shift_points[ind, :], shift_points if blur else RP, kernel)
			p_dist = np.linalg.norm(p_new - shift_points[ind, :], axis=1)
			still_shifting[ind] = p_dist >= atol
			shift_points[ind] = p_new
		if callback:
			callback(shift_points, it)
	return shift_points
