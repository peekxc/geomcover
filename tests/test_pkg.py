# %%
import numpy as np
from geomcover.cover import coverage, set_cover, to_canonical, valid_cover
from geomcover.io import load_set_cover


# %%
def test_setcover():
	subsets, weights = load_set_cover("mushroom")
	assert subsets.has_canonical_format, "Not in canonical format"

	for method in ["RR", "greedy", "ILP"]:
		soln, cost = set_cover(subsets, weights, method=method)
		cov = coverage(subsets, soln)
		assert len(cov) == subsets.shape[0]
		assert isinstance(cost, float) and cost == 22.0
		assert isinstance(soln, np.ndarray) and soln.size == subsets.shape[1]
		assert valid_cover(subsets, np.flatnonzero(soln))
		assert np.all(coverage(subsets, soln) == coverage(subsets, np.flatnonzero(soln)))


# https://archive.ics.uci.edu/dataset/73/mushroom

# def test_plotting():
# 	subsets, weights = load_set_cover("toy1")
# 	soln, cost = set_cover_ILP(subsets, weights)
# 	assert cost == 3
# 	from geomcover.plotting import plot_cover
# 	## TODO: get this working
# 	x, y = np.meshgrid([0, 1, 2, 3], [0, 1, 2])
# 	pos = np.c_[x.ravel(), y.ravel()]
# 	sets = sparse_to_sets(subsets)
# 	from bokeh.io import show
# 	show(plot_cover(sets, pos))
# 	from geomcover.cover import sets_to_sparse, sparse_to_sets
# 	sparse_to_sets(subsets)
# 	plot_cover()


def test_coverage():
	## From: https://optimization.cbe.cornell.edu/index.php?title=Set_covering_problem
	subsets, weights = load_set_cover("camera_stadium")
	for form in ["csr", "csc", "coo", "bsr", "lil"]:
		subsets = getattr(subsets, "to" + form)()
		assert np.all(coverage(subsets) == subsets.sum(axis=1))
		assert np.all(coverage(subsets, ind=[0, 1, 2]) == subsets.todense()[:, [0, 1, 2]].sum(axis=1))
		assert np.min(coverage(subsets)) <= 1
		soln, cost = set_cover(subsets, weights, "ilp")
		soln_ind = np.flatnonzero(soln)
		assert cost == 4
		assert np.min(coverage(subsets, soln_ind)) >= 1
		assert np.all(soln_ind == np.array([1, 2, 3, 4])) or np.all(soln_ind == np.array([0, 2, 4, 5]))


def test_medium():
	pass
