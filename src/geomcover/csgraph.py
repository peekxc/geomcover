import numpy as np
from itertools import islice, cycle, chain, combinations
from typing import Iterable, Generator, Sequence
from scipy.sparse import coo_array, sparray
from numbers import Integral

import geomcover
from geomcover import cover


def sliding_window(S: Iterable, n: int = 2) -> Generator:
	"""Generates a sliding window of width `n` from the iterable `S`.

	This function maps a m-length sequence `S` to a generator of tuples:

	  s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..., (s[m-n],s[m-n+1],...,sm)

	If the window size is larger than the iterable, the Generator will be empty.

	Parameters:
		S
	"""
	assert isinstance(n, Integral) and n >= 1
	it = iter(S)
	result = tuple(islice(it, n))
	if len(result) == n:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result


def cycle_window(S: Sequence, offset: int = 1, w: int = 2):
	"""Creates a cyclic windowed iterable over `S` with the given offset and width."""
	return sliding_window(islice(cycle(S), len(S) + offset), w)


def complete_graph(n: int):
	"""Creates the complete graph from the sequence [0,1,...,n-1]."""
	G = np.ones(shape=(n, n), dtype=bool)
	np.fill_diagonal(G, 0)
	return coo_array(G)


def cycle_graph(n: int, k: int = 2):
	"""Creates a cycle graph from the sequence [0,1,...,n-1], connecting all k-adjacent pairs (cyclically)."""
	S = np.fromiter(cycle_window(range(n), w=k, offset=k), dtype=(np.int32, k))
	E = np.fromiter(chain(*[combinations(s, 2) for s in S]), dtype=(np.int32, 2))
	E = np.unique(E, axis=0)
	# E = np.fromiter(chain(*[S[:,[i,j]] for i,j in pairwise(range(k))]), dtype=(np.int32, 2))
	G = coo_array((np.ones(len(E), dtype=bool), (E[:, 0], E[:, 1])), shape=(n, n))
	G = G + G.T
	return G


def path_graph(n: int, k: int = 2):
	"""Creates a path graph from the sequence [0,1,...,n-1], connecting all k-adjacent pairs."""
	S = np.fromiter(sliding_window(range(n), k), dtype=(np.int32, k))
	E = np.fromiter(chain(*[combinations(s, 2) for s in S]), dtype=(np.int32, 2))
	E = np.unique(E, axis=0)
	# E = np.fromiter(chain(*[S[:,[i,j]] for i,j in range(k)]), dtype=(np.int32, 2))
	G = coo_array((np.ones(len(E), dtype=bool), (E[:, 0], E[:, 1])), shape=(n, n))
	G = G + G.T
	return G


def nerve_complex(subsets: sparray, ind: np.ndarray, dim: int = 1):
	# subsets = M
	# ind = cover_ind
	from geomcover.cover import coverage, valid_cover

	## Use the definition of the laplacian to construct the edges
	assert valid_cover(subsets, ind), "Must be a valid cover"
	nerve_cover = subsets.tolil()[:, ind]
	nerve_laplacian = nerve_cover.T @ nerve_cover
	nerve_laplacian = nerve_laplacian.tocoo()

	from simplextree import SimplexTree

	st = SimplexTree()
	if dim >= 0:
		st.insert([[v] for v in range(len(ind))])
	if dim >= 1:
		st.insert(zip(nerve_laplacian.row, nerve_laplacian.col))
	if dim >= 2:
		nonempty_intersect = lambda s: bool((nerve_cover[:, np.array(s)] != 0).sum(axis=1).max() == len(s))
		st.expand(dim, nonempty_intersect)
	return st
