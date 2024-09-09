import array
import re
from typing import Any, Callable, Collection, Union, Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog
from scipy.sparse import csc_array, csc_matrix, issparse, sparray

from .io import _ask_package_install, sets_to_sparse, to_canonical

## All set cover algorithm implementations
SC_ALGS = ["rr", "greedy", "ilp", "sat"]

## All the mixed-integer linear program solvers, most of which come from or-tools
MIP_solvers = ["highs", "cbc", "glop", "bop", "sat", "scip", "gurobi_mip", "cplex_mip", "xpress_mip", "glpk_mip"]


def _validate_inputs(
	subsets: Union[np.ndarray, sparray], weights: Optional[ArrayLike] = None
) -> "tuple[sparray, np.ndarray]":
	if isinstance(subsets, Collection):
		subsets = sets_to_sparse(subsets)
	subsets: sparray = to_canonical(subsets, form="csc")
	assert issparse(subsets), "Cover must be sparse matrix"
	weights: np.ndarray = np.asarray(weights) if weights is not None else np.ones(subsets.shape[1], dtype=np.float64)
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	assert np.all(weights >= 0), "Set weights must be non-negative."
	return subsets, weights


## https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
## https://algnotes.info/on/obliv/lagrangian/set-cover-fractional/
## NOTE: This function originally used `sortednp` kway_merge, but this ended up being slower
def coverage(subsets: sparray, ind: Optional[np.ndarray] = None) -> np.ndarray:
	"""Returns the amount covered by each subset in the set of cover indices provided."""
	class_name = type(subsets).__name__
	if ind is not None:
		ind = np.flatnonzero(ind) if (len(ind) == subsets.shape[1] and ind.dtype == bool) else ind
	if "coo" in class_name:
		covered = np.zeros(subsets.shape[0], dtype=np.int64)
		mask = Ellipsis if ind is None else np.isin(subsets.col, ind)
		np.add.at(covered, subsets.row[mask], 1)
	elif "lil" in class_name:
		covered = subsets.sum(axis=1) if ind is None else subsets[:, ind].sum(axis=1)  # equiv, 5.9s
	else:
		z = np.zeros(subsets.shape[1], dtype=np.int64)
		z[ind] = 1
		covered = subsets @ z
	return covered


def valid_cover(subsets: sparray, ind: Optional[np.ndarray] = None) -> bool:
	"""Determines whether given sets form a *feasible* cover over the universe.

	Parameters:
		subsets: (n x J) sparse matrix describing `J` sets covering `n` elements.
		ind: index vector indicating which sets to check for cover feasibility. Defaults to all sets.

	Returns:
		a boolean indicating whether the subsets indexed by `ind` cover every element.
	"""
	return np.min(coverage(subsets, ind)) > 0


# Adapted from: http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
# Also see: https://courses.engr.illinois.edu/cs598csc/sp2011/Lectures/lecture_4.pdf
def _set_cover_greedy(subsets: Collection, weights: Optional[ArrayLike] = None):
	"""Approximates the weighted set cover problem via _greedy steps_.

	This function iteratively constructs a set cover by choosing the set that covers the largest number
	of yet uncovered elements with the least weight.

	Parameters:
		S: A collection of sets, represented as integer subsets
		W: non-negative weights for each subset

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.
	"""
	# assert issparse(subsets), "cover must be sparse matrix"
	# assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	# subsets = to_canonical(subsets, "csc", copy=True)
	weights = np.ones(len(subsets)) if weights is None else weights
	assert len(subsets) == len(weights), "Number of weights must match number of subsets"

	n, J = subsets.shape
	elements, cand_sets = set(range(n)), set(range(J))
	point_cover, set_cover = set(), set()  # array("I")

	## Make infinite costs finite, but very large
	weights = np.minimum(weights, 1.0 / np.finfo(float).resolution)

	## Map the sequences to sets
	subsets = list(map(subsets, set))

	## Greedily add the subsets with the most uncovered points
	while point_cover != elements:
		i = min(cand_sets, key=lambda j: np.inf if (p := len(subsets[j] - point_cover)) == 0.0 else weights[j] / p)
		set_cover.add(i)
		point_cover |= subsets[i]
		cand_sets -= set_cover
	assignment = np.zeros(J, dtype=bool)
	assignment[set_cover] = True
	return assignment, np.sum(weights[assignment])


def _grasp_heuristic(heuristic: str, seed: int | None = None) -> Callable[[ArrayLike], int]:
	if heuristic == "greedy":
		return np.argmin
	else:
		assert (
			isinstance(heuristic, str) and heuristic[:3] == "top"
		), f"Invalid heuristic '{heuristic}' supplied; must be one of ['greedy', 'topk', 'topk%'] where k >= 1."
		k = max(1, int(re.match("top(\\d+)[%]?", heuristic)[1]))
		if heuristic[-1] == "%":
			rng = np.random.default_rng(seed)

			def _heuristic(set_weights: np.ndarray) -> int:
				set_weights = np.asarray(set_weights)
				kth = max(1, min(np.ceil(len(set_weights) * (k / 100)), len(set_weights) - 1))
				indices = np.argpartition(set_weights, kth=kth)[:kth]
				prob = set_weights[indices] / np.sum(set_weights[indices])
				return rng.choice(indices, size=1, p=prob).item()

			return _heuristic
		else:
			rng = np.random.default_rng(seed)

			def _heuristic(set_weights: np.ndarray) -> int:
				set_weights = np.asarray(set_weights)
				kth = max(1, min(k, len(set_weights) - 1))
				indices = np.argpartition(set_weights, kth=kth)[:kth]
				prob = set_weights[indices] / np.sum(set_weights[indices])
				return rng.choice(indices, size=1, p=prob).item()

			return _heuristic

	# else:
	# 	raise ValueError(f"Invalid heuristic '{heuristic}' supplied; must be one of ['greedy', 'topk', 'topk%'] where k >= 1.")


def _maxsat_wcnf(subsets: csc_matrix, weights: np.ndarray):
	"""Generates a WMAX-SAT CNF formula from a weighted cover."""
	# assert isinstance(subsets, csc_matrix) or isinstance(subsets, csc_array), "Subset membership must be given as sparse (n x J) CSC array."
	subsets = to_canonical(subsets, "csc", copy=False)
	assert subsets.shape[1] == len(weights)

	from pysat.formula import WCNF

	n, J = subsets.shape

	## To encode covering constraints in CNF form, we use the transpose
	B = subsets.T.tocsc().sorted_indices()
	wcnf = WCNF()

	## (Hard) covering constraint: all elements must be covered
	N = [z for z in np.split(B.indices + 1, B.indptr)[1:-1]]
	wcnf.extend(N, weights=None)

	## (Soft) weight constraint: encourage less subsets by accumulating negative variables
	wcnf.extend(-(np.arange(J) + 1)[:, np.newaxis], weights=list(weights))

	## return valid formula
	return WCNF(from_string=wcnf.to_dimacs())


# Greedy provides an H_k-approximation for the weight k set cover, where H_k = \sum\limits_{i=1}^k 1/i is the k-th harmonic number
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.1615&rep=rep1&type=pdf
# See also concise bounds w/ references: https://www.cs.ucr.edu/~neal/Young08SetCover.pdf
# Nice slides: http://cs.williams.edu/~shikha/teaching/spring20/cs256/lectures/Lecture31.pdf
# NOTE: Attempted optimization of v @ A for sparse v via the following:
# 	to_count = np.isin(subsets.indices, np.flatnonzero(1 - point_cover))
# 	set_counts = np.maximum(np.add.reduceat(to_count, subsets.indptr[:-1]), 1/n)
# Though it works, it's roughly 5x slower
def set_cover_greedy(subsets: sparray, weights: Optional[ArrayLike] = None) -> tuple:
	r"""Approximates the weighted set cover problem via _greedy steps_.

	This function iteratively constructs a set cover by choosing the set that covers the largest number
	of yet uncovered elements with the least weight.

	The greedy strategy is a very fast SC algorithm, though counter-examples have demonstrated the method
	can produce poor covers on certain pathological inputs. It has been shown that the algorithm has a
	worst-case multiplicative $\log(n + 1)$-approximation factor [1].

	Parameters:
		subsets: (n x J) sparse matrix of ``J`` subsets whose union forms a cover over ``n`` points.
		weights: (J)-length array of subset weights.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.

	Notes:
		The algorithm implemented here uses the 'dual-fitting' variant discussed in 5.3 of [2] below, which \
		can be used used to generate a feasible solution to dual LP. 

	References:
		1. Feige, Uriel. "A threshold of ln n for approximating set cover." Journal of the ACM (JACM) 45.4 (1998): 634-652.
		2. [CS 583 notes by Chandra Chekuri](https://courses.grainger.illinois.edu/cs583/sp2018/Notes/covering.pdf)
	"""
	subsets, weights = _validate_inputs(subsets, weights)
	n, J = subsets.shape
	slice_col = lambda j: subsets.indices[slice(*subsets.indptr[[j, j + 1]])]  # noqa: E731

	## Vectorized version of the set version that uses matrix multiplication
	set_counts = subsets.sum(axis=0)
	point_cover, soln = np.zeros(n, dtype=bool), np.zeros(J, dtype=bool)
	while not np.all(point_cover):
		opt_s = np.argmin(weights / set_counts)
		point_cover[slice_col(opt_s)] = True
		set_counts = np.maximum((1 - point_cover) @ subsets, 1 / n)
		soln[opt_s] = True
	return soln, np.sum(weights[soln])


def set_cover_rr(
	subsets: sparray,
	weights: Optional[ArrayLike] = None,
	maxiter: Optional[int] = None,
	sparsity: float = 1.0,
	seed: Optional[int] = None,
) -> tuple:
	r"""Approximates the weighted set cover problem via _randomized rounding_.

	This function first computes a minimum-cost fractional set cover whose solution lower-bounds the optimal solution,
	then uses randomized rounding to produce a sequence of solutions whose objectives slightly increase this bound,
	continuing until a feasible solution is found.

	The minimum-cost fractional cover is obtained by solving the following linear program:

	$$\begin{align*}\text{minimize} \quad & \sum\limits_{j \in C} s_j \cdot w_j  \\
	\text{s.t.} \quad & \sum\limits_{j \in N_i}  s_j  \geq 1, \quad \forall \, i \in [n] \\
	& s_j \in [0, 1], \quad \forall \, j \in [J]\end{align*}$$

	where $s_j \in [0, 1]$ is a real number indicating the strength of the membership $S_j \in \mathcal{S}$ and $N_i$
	represents the subsets of $S$ that the element $x_i$ intersects. The randomized rounding procedure iteratively adds
	sets $S_j$ with probability $c \cdot s_j$ until a feasible cover is found.

	If not supplied, `maxiter` defaults to $(2 / c) \log(n)$ where $c$ is given by the `sparsity` argument.
	Supplying `sparsity` values lower than 1 allows choosing fewer subsets per iteration, which can
	result in sparser or lower weight covers at the cost of more iterations.

	Parameters:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		maxiter: number of iterations to repeat the sampling process. See details.
		sparsity: constant used to emphasize sparsity between (0, 1]. See details.
		seed: seed for the random number generator. Use an integer for deterministic computation.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.

	See Also:
		- [Sparse arrays](`scipy.sparse`)
		- [reference](/index.qmd)

	Notes:
		This function requires `subsets` to be a sparse matrix in [canonical](https://docs.scipy.org/doc/scipy/tutorial/sparse.html#canonical-formats) CSC form. \
		If `subsets` is not in this form, a copy of the `subsets` is converted first; to avoid this for maximum performance, ensure the subset \ 
		matrix is in canonical form first.

	Examples:
		```{python}
		from geomcover.cover import set_cover_rr
		from geomcover.io import load_set_cover
		
		subsets, weights = load_set_cover("mushroom")
		soln, cost = set_cover_rr(subsets, weights)
		n, J = subsets.shape

		print("Set family with {n} elements and {J} sets can be covered with {np.sum(soln)} sets with cost {cost}.")
		```
	"""
	## Always convert to a canonical format first
	subsets, weights = _validate_inputs(subsets, weights)
	n, J = subsets.shape

	## Convert input types as necessary
	W = weights.reshape((1, len(weights)))  # ensure W is a column vector
	if subsets.dtype != int:
		subsets = subsets.astype(int)

	## Since A_ub @ x <= b_ub, negate A_ub to enable constraint A_ub @ x >= 1
	b_ub = np.repeat(-1.0, n)
	soln = linprog(c=W, b_ub=b_ub, A_ub=-subsets, bounds=(0.0, 1.0))  # options={"sparse": True}
	assert soln.success, "Linear progam did not find a solution, which suggests problem is infeasible"

	## Change maxiter to default solution
	maxiter = int(np.ceil((2 / sparsity) * np.log(n))) if maxiter is None else int(maxiter)

	## Sample subsets elementwise with probability 'p_i' until achieving a valid cover.
	## Repeat the sampling process 'maxiter' times and choose the best one
	rng = np.random.default_rng(seed=seed)
	p = soln.x
	t = sparsity * p
	best_cost, assignment = np.inf, np.zeros(len(p), dtype=bool)
	for _ in range(maxiter):
		z = rng.random(len(p)) <= t
		while np.any((subsets @ z) < 1.0):  # while any point is left uncovered
			z = np.logical_or(z, rng.random(len(p)) <= t)
		cost = np.dot(W, z)
		best_cost, assignment = (cost, z) if cost < best_cost else (best_cost, assignment)
	return assignment, np.ravel(best_cost).item()


def set_cover_ilp(subsets: sparray, weights: Optional[ArrayLike] = None, solver: str = "highs") -> tuple:
	r"""Approximates the weighted set cover problem via _integer linear programming_.

	This function attempts to directly solve weighted set cover problem by reducing it to 
	the following mixed-integer linear program:

	$$\begin{align*}\text{minimize} \quad & \sum\limits_{j \in C} s_j \cdot w_j  \\
	\text{s.t.} \quad & \sum\limits_{j \in N_i}  s_j  \geq 1, \quad \forall \, i \in [n] \\
	& s_j \in \{0, 1\}, \quad \forall \, j \in [J]\end{align*}$$

	where $s_j \in \{0, 1\}$ is a indicator of set membership $S_j \in \mathcal{S}$ and $N_i$
	represents the subsets of $S$ that the element $x_i$ intersects. The algorithm is deterministic,
	and it typically finds the global optimum of moderately challenging mixed-integer linear programs
	(when it exists).

	Parameters:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		solver: which MILP solver to use. Defaults to the HiGHS solver in SciPy.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.

	See Also:
		- [Mixed-integer program](scipy.optimize.milp)
	"""
	subsets, weights = _validate_inputs(subsets, weights)
	assert solver.lower() in MIP_solvers, f"Unknown solver supplied '{solver}'; must be one of {str(MIP_solvers)}"
	if solver == "highs":
		from scipy.optimize import LinearConstraint, milp  # noqa: PLC0415

		subsets = to_canonical(subsets, "csc", copy=True)
		c = np.ravel(weights).astype(np.float64)
		b_u = -np.ones(subsets.shape[0])
		b_l = np.full_like(b_u, -np.inf, dtype=float)
		subsets.data = np.negative(subsets.data.astype(np.float64, copy=False))
		subsets.indices = subsets.indices.astype(np.int32, copy=False)
		subsets.indptr = subsets.indptr.astype(np.int32, copy=False)
		constraints = LinearConstraint(subsets, lb=b_l, ub=b_u)
		integrality = np.ones_like(c)
		res = milp(c=c, constraints=constraints, integrality=integrality)
		assert res.success, res.message
		return res.x.astype(bool), res.fun
	else:
		_ask_package_install("ortools")
		from ortools.linear_solver import pywraplp  # noqa: PLC0415

		## Choose the solver
		solver = pywraplp.Solver.CreateSolver(solver.upper())  # mip solver
		assert solver is not None

		## Setup the constraints; note we convert to CSR for contiguous stride indexing
		B = to_canonical(subsets, form="csr", copy=True)
		subset_indicators = [solver.IntVar(0, 1, "") for i in range(B.shape[1])]
		min_weight_obj = solver.Sum([s * w for s, w in zip(subset_indicators, weights)])
		for z in np.split(B.indices, B.indptr)[1:-1]:
			solver.Add(solver.Sum([subset_indicators[zi] for zi in z]) >= 1)

		## Call the solver
		solver.Minimize(min_weight_obj)
		status = solver.Solve()
		assert status in {0, 1}, "Failed to find a feasible solution"

		## Extract solution and cost
		soln = np.array([s.solution_value() for s in subset_indicators], dtype=bool)
		min_cost = solver.Objective().Value()
		return soln, min_cost


def set_cover_sat(
	subsets: sparray, weights: Optional[ArrayLike] = None, full_output: bool = False, **kwargs: dict
) -> tuple:
	"""Computes an approximate solution to the weighted set cover problem via *weighted MaxSAT*.

	This function converts the problem of finding a minimal weight set cover to a weighted MaxSAT instance, \
	which is known to achieve at least an (8/7)-approximation of the optimal solution.

	The default solver is the Relaxed Cardinality Constraint solver (RC2), with Boolean lexicographic optimization \
	(BLO) stratification options turned on. RC2 requires `python-sat` to be installed.

	Parameters:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		full_output: whether to return the SAT-solver instance. Defaults to False.
		**kwargs: additional keyword arguments to pass to the solver.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.
	"""
	_ask_package_install("pysat")
	subsets, weights = _validate_inputs(subsets, weights)
	# assert "examples" in dir(pysat), "Please install the full pysat package with extensions for SAT support"
	from pysat.examples.rc2 import RC2Stratified  # noqa: PLC0415

	wcnf = _maxsat_wcnf(subsets, weights)
	solver = RC2Stratified(wcnf, **kwargs)
	finished: bool = False
	clauses: np.ndarray = np.empty(shape=(0,), dtype=np.int64)
	try:
		clauses = np.array(solver.compute())
		finished = True
	except KeyboardInterrupt:
		finished = False
	finally:
		set_ind = clauses[clauses > 0] - 1 if finished and len(clauses) > 0 else np.abs(np.array(solver.core_sels)) - 1
		assignment = np.zeros(subsets.shape[1], dtype=bool)
		assignment[set_ind] = True
		out = (assignment, np.sum(weights[assignment]))
	return out if not full_output else out, dict(solver=solver, formula=wcnf, clauses=clauses)


def set_cover(subsets: sparray, weights: Optional[ArrayLike] = None, method: str = "RR", **kwargs: dict) -> tuple:  # type: ignore
	"""Computes an approximate solution to the weighted set cover problem.

	This is essentially a lightweight wrapper around the various set cover implementations, which can be configured via
	the `method` argument (supported options are {'RR', 'GREEDY', 'ILP', 'SAT'}). All additional keyword-arguments are
	forwarded to their subsequent solvers.

	Parameters:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		**kwargs: additional keyword arguments to pass to the solver.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.
	"""
	method = str(method).upper()
	if method == "RR":
		return set_cover_rr(subsets, weights, **kwargs)  # type: ignore
	elif method == "GREEDY":
		return set_cover_greedy(subsets, weights, **kwargs)  # type: ignore
	elif method == "ILP":
		return set_cover_ilp(subsets, weights, **kwargs)  # type: ignore
	elif method == "SAT":
		return set_cover_sat(subsets, weights, **kwargs)  # type: ignore
	else:
		raise ValueError(f"Invalid method '{method}' supplied; must be one of {SC_ALGS}")
