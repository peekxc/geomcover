"""I/O related functions for loading set cover instance problems or converting them to canonical formats."""

import importlib
import re
from array import array
from importlib import resources
from typing import Collection, Iterable
from urllib.request import urlopen

import numpy as np
from scipy.sparse import coo_array, csc_array, issparse, sparray


def to_canonical(A: sparray, form: str = "csc", copy: bool = False) -> sparray:
	"""Converts a sparse array into a supplied form, gauranteeing canonical format.
	
	This function converts a given SciPy sparse array into a canonical form respecting [has_canonical_format]().
	Here, a sparse array `A` is said to be in *canonical form* if all of the properties below hold (where relevent):
		1. Indices are sorted in non-descending order, where eligible.
		2. Zero entries are removed.
		3. Duplicate entries are merged (via summation).
		4. Padding between strides is pruned. 
	If `A` is in canonical format to begin with, it is returned unmodified. Otherwise, `A` is modified in-place 
	and its reference is returned, unless `copy=False`.  

	Parameters:
		A: sparse array.
		form: target form to convert `A`, such as "csc", "csr", "coo", etc.
		copy: whether to return a copy of array. By default, `A` is modified by reference. 

	Returns:
		sparse array in canonical form. 
	"""
	assert isinstance(form, str) and form.lower() in {'csc', 'csr', 'lil', 'dok', 'coo'}, f"Invalid form '{form}'; must be a format supported by SciPy."  # fmt: skip
	A = getattr(A, "to" + form)()
	if hasattr(A, "has_sorted_indices"):
		## https://stackoverflow.com/questions/28428063/what-does-csr-matrix-sort-indices-do
		A.has_sorted_indices = False
	for clean_op in ["sort_indices", "eliminate_zeros", "sum_duplicates", "prune"]:
		if hasattr(A, clean_op):
			getattr(A, clean_op)()
	assert A.has_canonical_format, "Failed to convert sparse matrix to canonical form"
	return A.copy() if copy else A


def sets_to_sparse(S: Collection, reindex: bool = False, form: str = "csc") -> csc_array:
	r"""Converts a collection of sets into a sparse array.

	This function converts a `Collection` of integer-valued sequences into a sparse matrix, where
	each column represents a set and each row represents an element. Optionally, if
	the sets themselves do not represent 0-based indices, you can set `reindex=True`
	to map the elements to the base index set $[n] = {0, 1, ..., n - 1}$.

	Parameters:
		S: Collection of indices representing subsets.
		reindex: whether to reindex the sets to the base index set. Default to False.

	Returns:
		sparse boolean array in canonical form.
	"""
	indptr = np.zeros(len(S) + 1, dtype=np.int64)
	indptr[1:] = [len(s) for s in S]
	N_ELEMS = np.sum(indptr)
	np.cumsum(indptr, out=indptr)

	## Speedy alternative to np.concatenate([list(s) for s in S])
	indices = np.empty(N_ELEMS, dtype=np.int64)
	for s, i, j in zip(S, indptr[:-1], indptr[1:], strict=True):
		indices[i:j] = np.fromiter(s, count=j - i, dtype=np.int64)
	data = np.ones(np.size(indices), dtype=bool)

	## Re-index to the base index set if requested
	if reindex:
		keys, indices = np.unique(indices, return_inverse=True)
		n = np.size(keys)
	else:
		n = np.max(indices) + 1
	A = csc_array((data, indices, indptr), shape=(n, np.size(indptr) - 1), dtype=bool)
	A = to_canonical(A, form, copy=False)
	return A


def reindex_sparse(subsets: sparray) -> None:
	"""Reindexes the indices of a given sparse array to the base index set."""
	class_name = type(subsets).__name__.lower()
	if "coo" in class_name:
		subsets.row = np.unique(subsets.row, return_inverse=True)[1]
		subsets.col = np.unique(subsets.col, return_inverse=True)[1]
	elif "csc" in class_name or "csr" in class_name:
		subsets.indices = np.unique(subsets.indices, return_inverse=True)[1]
	else:
		raise NotImplementedError("Haven't implemented reindexing from non-COO, CSR, or CSC arrays.")


def sparse_to_sets(subsets: sparray, reindex: bool = False) -> csc_array:
	r"""Converts a collection of sets into a sparse CSC array.

	This function converts a a sparse matrix into list of integer-valued arrays, where
	each array represents a subset. Optionally, if the sets themselves do not represent
	0-based indices, you can set `reindex=True` to map the rows and columns to the base index
	set $[n] = {0, 1, ..., n - 1}$.

	Parameters:
		subsets: sparse array in canonical form.
		reindex: whether to reindex the sets to the base index set. Default to False.

	Returns:
		Collection of indices representing subsets.
	"""
	if hasattr(subsets, "row") and hasattr(subsets, "col"):
		if reindex:
			reindex_sparse(subsets)
		return [subsets.row[subsets.col == j] for j in range(subsets.shape[1])]
	elif "csc" in type(subsets).__name__.lower():
		if reindex:
			reindex_sparse(subsets)
		return np.split(subsets.indices, subsets.indptr[1:-1])
	else:
		assert issparse(subsets), "Must be a sparse array"
		return sparse_to_sets(subsets.tocsc())


## J. E. Beasley's collection of test data sets for the OR-lib
## From: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
# fmt: off
OR_TEST_FILES = [
	'scp41', 'scp42', 'scp43', 'scp44', 'scp45', 'scp46', 'scp47', 'scp48', 'scp49', 'scp410',
	'scp51', 'scp52', 'scp53', 'scp54', 'scp55', 'scp56', 'scp57', 'scp58', 'scp59', 'scp510',
	'scp61', 'scp62', 'scp63', 'scp64', 'scp65',
	'scpa1', 'scpa2', 'scpa3', 'scpa4', 'scpa5',
	'scpb1', 'scpb2', 'scpb3', 'scpb4', 'scpb5',
	'scpc1', 'scpc2', 'scpc3', 'scpc4', 'scpc5',
	'scpd1', 'scpd2', 'scpd3', 'scpd4', 'scpd5',
	'scpe1', 'scpe2', 'scpe3', 'scpe4', 'scpe5'
]

## From: https://optimization.cbe.cornell.edu/index.php?title=Set_covering_problem
CAMERA_STADIUM = np.array([1,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,0,0])

## From: https://www.gcardone.net/2019-08-31-visa-free-travel/
TOYSET1 = [
	[0,1,4,5,8,9],
	[5,6,9,10],
	[8,9,10,11],
	[2,4,5,6,7],
	[3,7],
	[0,1,2,3]
]

# fmt: on


## To load data sets from: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
def load_set_cover(test_set: str) -> tuple:
	"""Loads an instance of for testing weighted set cover algorithms.

	Parameters:
		test_set: name of the available test sets. See details.

	Test Sets:
		The following test sets are available for testing.
		- 'toy':
		- 'camera_stadium':
		- 'mushroom': https://archive.ics.uci.edu/dataset/73/mushroom
		- 'scp*': Set cover problem instance from OR library.

	"""
	if test_set.lower() in {"camera_stadium", "camera stadium"}:
		A = csc_array(CAMERA_STADIUM.reshape((15, 8)))
		set_weights = np.ones(A.shape[1])
	elif test_set.lower() == "toy1":
		A = sets_to_sparse(TOYSET1, reindex=True)
		set_weights = np.ones(A.shape[1])
	elif test_set.lower() == "mushroom":
		with resources.path("geomcover.data", "mushroom.dat") as fn:
			sets = np.loadtxt(fn)
			A = sets_to_sparse(sets, reindex=True)
			set_weights = np.ones(A.shape[1])
	elif test_set.lower() in OR_TEST_FILES:
		data = urlopen(f"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{test_set.lower()}.txt")
		lines = data.readlines()
		lines = [ln.decode() for ln in lines]
		num_elements, num_sets = map(int, lines[0].split())
		data = "".join(lines[1:]).replace("\n ", "")
		data = np.array([float(i) for i in re.split(r"\s+", data) if len(i) > 0])
		set_weights = data[:num_sets]
		data = data[num_sets:]
		cc = 0
		R, C = array("I"), array("I")
		for j in range(num_elements):
			nj = int(data[cc])
			row_j = data[(cc + 1) : (cc + nj + 1)]
			cc += nj + 1
			C.extend(row_j.astype(int) - 1)
			R.extend(j * np.ones(nj, dtype=int))
		A = coo_array((np.ones(len(R)), (R, C)), shape=(num_elements, num_sets), dtype=bool)
		A.eliminate_zeros()
		A.sum_duplicates()
	else:
		raise ValueError(f"Unknown test data set '{test_set}'; must be a set cover instance")
	return A, set_weights


def _package_exists(package: str) -> bool:
	"""Checks whether a package exists via importlib."""
	pkg_spec = importlib.util.find_spec(package)
	return pkg_spec is not None


def _ask_package_install(package: str) -> None:
	"""Checks whether a package exists via importlib, and if not raises an exception."""
	if not (_package_exists(package)):
		raise RuntimeError(f"Module {package} not installed. To use this function, please install {package}.")
