import importlib
import numpy as np
import re
from typing import Callable
from numpy.typing import ArrayLike
from scipy.sparse import issparse, csc_matrix, csc_array
from scipy.optimize import linprog

from .loaders import to_canonical

## All the mixed-integer linear program solvers, most of which come from or-tools
MIP_solvers = ["highs", "cbc", "glop", "bop", "sat", "scip", "gurobi_mip", "cplex_mip", "xpress_mip", "glpk_mip"]

def package_exists(package: str) -> bool: 
	"""Checks whether a package exists via importlib."""
	pkg_spec = importlib.util.find_spec(package)
	return(pkg_spec is not None)

def ask_package_install(package: str):
	"""Checks whether a package exists via importlib, and if not raises an exception."""
	if not(package_exists(package)):
		raise RuntimeError(f"Module {package} not installed. To use this function, please install {package}.")

## https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
## https://algnotes.info/on/obliv/lagrangian/set-cover-fractional/
def coverage(subsets: csc_array, ind: np.ndarray = None) -> np.ndarray:
	"""Returns the amount covered by each subset in the set of cover indices provided."""
	# A.tolil()[:,np.flatnonzero(soln)].sum(axis=1) # equiv, 5.9s
	# lili approach: 1.4s
	subsets = to_canonical(subsets, "coo")
	covered = np.zeros(subsets.shape[0], dtype=np.int64)
	if ind is None:
		np.add.at(covered, subsets.row, 1)
	else: 
		np.add.at(covered, subsets.row[np.isin(subsets.col, ind)], 1)
	return covered

def valid_cover(A: csc_array, ind: np.ndarray = None) -> bool:
	"""Determines whether given sets form a valid or *feasible* cover over the universe."""
	return np.min(coverage(A, ind)) > 0

	# n, J = A.shape
	# subset_splits = np.split(A.indices, A.indptr)[1:-1]
	# assert len(subset_splits) == J, "Splitting of cover array failed. Are there empty columns?"
	# if ind is not None:
	# 	ind = np.array(ind).astype(int) 
	# 	subset_splits = [subset_splits[i] for i in ind]
	# covered_ind = sortednp.kway_merge(*subset_splits, assume_sorted=True, duplicates=4)
	# return len(covered_ind) == n

def wset_cover_RR(subsets: ArrayLike, weights: ArrayLike, maxiter: int = "default", sparsity: float = 1.0, seed = None) -> tuple:
	"""Computes an approximate solution to the weighted set cover problem via randomized rounding.

	This function first computes a minimum-cost fractional set cover whose solution lower-bounds the optimal solution, 
	then uses randomized rounding to produce a sequence of solutions whose objectives slightly increase this bound, 
	continuing until a feasible solution is found. 

	If not supplied, maxiter defaults to (2 / sparsity) * log(n). Supplying 'sparsity' values lower than 1 
	will result in choosing fewer subsets per iteration, which may result in sparser and/or lower 
	weight cover (at the cost of more iterations). 

	Args:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		maxiter: number of iterations to repeat the sampling process. See details.  
		sparsity: constant used to emphasize sparsity between (0, 1]. See details. 

	Returns:
		tuple: (s, c) where s is a boolean array indicating which subsets are in the cover and c is its cost.

	Examples:
		```{python}
			1 + 1
		```
	"""
	assert issparse(subsets), "Cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	weights = np.asarray(weights)
	assert np.all(weights >= 0), "Set weights must be non-negative."
	
	## Always convert to a canonical format first
	subsets = to_canonical(subsets, "csc")

	## Convert input types as necessary
	W = weights.reshape((1, len(weights))) # ensure W is a column vector
	if subsets.dtype != int:
		subsets = subsets.astype(int)

	## Since A_ub @ x <= b_ub, negate A_ub to enable constraint A_ub @ x >= 1
	b_ub = np.repeat(-1.0, subsets.shape[0])
	soln = linprog(c=W, b_ub=b_ub, A_ub=-subsets, bounds=(0.0, 1.0)) # options={"sparse": True}
	assert soln.success, "Linear progam did not find a solution, which suggests problem is infeasible"

	## Change maxiter to default solution
	if maxiter == "default":
		maxiter = int(np.ceil((2/sparsity) * np.log(subsets.shape[0])))

	## Sample subsets elementwise with probability 'p_i' until achieving a valid cover.
	## Repeat the sampling process 'maxiter' times and choose the best one
	rng = np.random.default_rng(seed=seed)
	p = soln.x
	t = sparsity * p
	best_cost, assignment = np.inf, np.zeros(len(p), dtype=bool)
	for _ in range(maxiter):
		z = rng.random(len(p)) <= t
		while np.any((subsets @ z) < 1.0): # while any point is left uncovered
			z = np.logical_or(z, rng.random(len(p)) <= t)
		cost = np.dot(W, z)
		best_cost, assignment = (cost, z) if cost < best_cost else (best_cost, assignment)
	return assignment, np.ravel(best_cost).item()

# Adapted from: http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
# Also see: https://courses.engr.illinois.edu/cs598csc/sp2011/Lectures/lecture_4.pdf
# S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
# 		 s.t. union(S_1, ..., S_k) covers [n]
# def wset_cover_greedy(subsets: csc_matrix, weights: ArrayLike, ):
# 	"""
# 	Computes a set of indices I whose subsets S[I] = { S_1, S_2, ..., S_k }
# 	yield an approximation to the minimal weighted set cover, i.e.
	
# 	Parameters: 
# 		S: An (n x m) sparse matrix whose non-zero elements indicate subset membership (one subset per column)
# 		W: Weights for each subset 

# 	Returns: 
# 		(s, c) := tuple where s is a boolean vector indicating which subsets are included in the optimal solution 
# 		and c is the minimized cost of that solution. 
# 	"""
# 	assert issparse(subsets), "cover must be sparse matrix"
# 	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
# 	subsets = to_canonical(subsets, "csc", copy=True)

# 	S, W = subsets, weights
# 	n, J = S.shape
# 	elements, sets, point_cover, set_cover = set(range(n)), set(range(J)), set(), array('I')
# 	slice_col = lambda j: S.indices[S.indptr[j]:S.indptr[j+1]] # provides efficient column slicing

# 	## Make infinite costs finite, but very large
# 	if np.any(W == np.inf):
# 		W[W == np.inf] = 1.0/np.finfo(float).resolution

# 	## Greedily add the subsets with the most uncovered points
# 	while point_cover != elements:
# 		#I = min(sets, key=lambda j: W[j]/len(set(slice_col(j)) - point_cover) ) # adding RHS new elements to cover incurs weighted cost of w/|RHS|
# 		I = min(sets, key=lambda j: np.inf if (p := len(set(slice_col(j)) - point_cover)) == 0.0 else W[j]/p)
# 		set_cover.append(I)
# 		point_cover |= set(slice_col(I))
# 		sets -= set(set_cover)
# 	assignment = np.zeros(J, dtype=bool)
# 	assignment[set_cover] = True
# 	return (assignment, np.sum(weights[assignment])) if not info else set_cover


def _grasp_heuristic(heuristic: str, seed: int | None = None) -> Callable[[ArrayLike], int]:
	if heuristic == "greedy":
		return np.argmin
	else:
		assert isinstance(heuristic, str) and heuristic[:3] == "top", f"Invalid heuristic '{heuristic}' supplied; must be one of ['greedy', 'topk', 'topk%'] where k >= 1."
		k = max(1, int(re.match("top(\\d+)[%]?", heuristic)[1]))
		if heuristic[-1] == "%":
			rng = np.random.default_rng(seed)
			def _heuristic(set_weights: np.ndarray) -> int:
				set_weights = np.asarray(set_weights)
				kth = max(1, min(np.ceil(len(set_weights) * (k / 100)), len(set_weights)-1))
				indices = np.argpartition(set_weights, kth=kth)[:kth]
				prob = set_weights[indices] / np.sum(set_weights[indices])
				return rng.choice(indices, size=1, p=prob).item()
			return _heuristic
		else:
			rng = np.random.default_rng(seed)
			def _heuristic(set_weights: np.ndarray) -> int:
				set_weights = np.asarray(set_weights)
				kth = max(1, min(k, len(set_weights)-1))
				indices = np.argpartition(set_weights, kth=kth)[:kth]
				prob = set_weights[indices] / np.sum(set_weights[indices])
				return rng.choice(indices, size=1, p=prob).item()
			return _heuristic

	# else:
	# 	raise ValueError(f"Invalid heuristic '{heuristic}' supplied; must be one of ['greedy', 'topk', 'topk%'] where k >= 1.")

## Note: Attempted optimization of v @ A for sparse v via the following: 
# to_count = np.isin(subsets.indices, np.flatnonzero(1 - point_cover))
# set_counts = np.maximum(np.add.reduceat(to_count, subsets.indptr[:-1]), 1/n)
# Though it works, it's roughly 5x slower
def wset_cover_greedy(subsets: csc_matrix, weights: ArrayLike) -> tuple:
	"""Computes an approximate solution to the weighted set cover problem via a greedy approach.
	
	Parameters:
		subsets: (n x J) sparse matrix of ``J`` subsets whose union forms a cover over ``n`` points.
		weights: (J)-length array of subset weights.

	Returns:
		pair (s, c) where ``s`` is an array indicating cover membership and ``c`` its cost.
	"""
	assert issparse(subsets), "cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	subsets = to_canonical(subsets, "csc")
	n, J = subsets.shape
	slice_col = lambda j: subsets.indices[slice(*subsets.indptr[[j,j+1]])] # noqa: E731
	
	## Vectorized version of the set version that trac
	set_counts = subsets.sum(axis=0)
	point_cover, soln = np.zeros(n, dtype=bool), np.zeros(J, dtype=bool)
	while not np.all(point_cover):
		opt_s = np.argmin(weights / set_counts)
		point_cover[slice_col(opt_s)] = True
		set_counts = np.maximum((1 - point_cover) @ subsets, 1/n)
		soln[opt_s] = True
	return soln, np.sum(weights[soln])

def _maxsat_wcnf(subsets: csc_matrix, weights: ArrayLike):
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
	N = [z for z in np.split(B.indices+1, B.indptr)[1:-1]]
	wcnf.extend(N, weights=None)

	## (Soft) weight constraint: encourage less subsets by accumulating negative variables 
	wcnf.extend(-(np.arange(J)+1)[:,np.newaxis], weights=list(weights))
	
	## return valid formula
	return WCNF(from_string=wcnf.to_dimacs())

# for j in range(subsets.shape[1]): 
# 	wcnf.append(list(np.flatnonzero(subsets[:,j].A)+1), weight=None)
# for subset in np.split(subsets.indices, subsets.indptr)[1:-1]:
# 	wcnf.append(list(subset+1), weight=None)
# for j, w in enumerate(weights): 
# 	wcnf.append([-int(j+1)], weight=w)
## The types are not inferred from direct inputs, so have to write to file to get correct parsing
# tmp = NamedTemporaryFile()
# wcnf.to_file(tmp.name)
# wcnf = WCNF(from_file=tmp.name)
# tmp.close()


def wset_cover_sat(subsets: csc_matrix, weights: ArrayLike, return_solver: bool = False, **kwargs) -> tuple:
	"""Computes an approximate solution to the weighted set cover problem via *weighted MaxSAT*.

	This function converts the problem of finding a minimal weight set cover to a weighted MaxSAT instance, 
	which is known to achieve at least an (8/7)-approximation of the optimal solution. The solver instance
	used here is the Relaxed Cardinality Constraint solver (RC2) with Boolean lexicographic optimization (BLO)
	and stratification (which requires python-sat to be installed).

	Args:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		return_solver: whether to return the SAT-solver instance. Defaults to False. 
		**kwargs: additional keyword arguments to pass to the solver. 

	Returns:
		tuple: (s, c) where s is a boolean array indicating which subsets are in the cover and c is its cost.
	"""	
	ask_package_install("pysat")
	# assert "examples" in dir(pysat), "Please install the full pysat package with extensions for SAT support"
	from pysat.examples.rc2 import RC2Stratified
	subsets = to_canonical(subsets, "csc", copy=True)
	wcnf = _maxsat_wcnf(subsets, weights)
	solver = RC2Stratified(wcnf, **kwargs)
	finished, clauses = False, None
	try: 
		clauses = np.array(solver.compute())
		finished = True
	except KeyboardInterrupt:
		finished = False
	finally: 
		set_ind = clauses[clauses > 0]-1 if finished else np.abs(np.array(solver.core_sels))-1 # 0-based
		if (return_solver):
			return(solver)
		assignment = np.zeros(subsets.shape[1], dtype=bool)
		assignment[set_ind] = True
		return(assignment, np.sum(weights[assignment]))

def wset_cover_ILP(subsets: csc_matrix, weights: ArrayLike, solver: str = "highs") -> tuple:
	"""Computes an approximate solution to the weighted set cover problem via *integer linear programming*.

	Args:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points.
		weights: (J)-length array of subset weights.
		solver: which MILP solver to use. Defaults to the HiGHS solver in SciPy.

	Returns:
		tuple: (s, c) where s is a boolean array indicating which subsets are in the cover and c is its cost.
	""" 
	assert solver.lower() in MIP_solvers, f"Unknown solver supplied '{solver}'; must be one of {str(MIP_solvers)}"
	if solver == "highs":
		from scipy.optimize import LinearConstraint, milp
		subsets = to_canonical(subsets, "csc", copy=True)
		c = np.ravel(weights).astype(np.float64)
		b_u = -np.ones(subsets.shape[0])
		b_l = np.full_like(b_u, -np.inf, dtype=float)
		subsets.data = np.negative(subsets.data.astype(np.float64, copy=False))
		subsets.indices	= subsets.indices.astype(np.int32, copy=False)
		subsets.indptr = subsets.indptr.astype(np.int32, copy=False)
		constraints = LinearConstraint(subsets, lb=b_l, ub=b_u)
		integrality = np.ones_like(c)
		res = milp(c=c, constraints=constraints, integrality=integrality)
		assert res.success, res.message
		return res.x.astype(bool), res.fun
	else:
		ask_package_install("ortools")
		from ortools.linear_solver import pywraplp
		
		## Choose the solver 
		solver = pywraplp.Solver.CreateSolver(solver.upper()) # mip solver
		assert solver is not None
		
		## Setup the constraints 
		B = to_canonical(subsets, form="csr", copy=True)
		subset_indicators = [solver.IntVar(0,1,"") for i in range(B.shape[1])]
		min_weight_obj = solver.Sum([s*w for s, w in zip(subset_indicators, weights)])
		for z in np.split(B.indices, B.indptr)[1:-1]:
			solver.Add(solver.Sum([subset_indicators[zi] for zi in z]) >= 1)
		
		## Call the solver 
		solver.Minimize(min_weight_obj)
		status = solver.Solve()
		assert status == 0 or status == 1, "Failed to find a feasible solution"
		
		## Extract solution and cost  
		soln = np.array([s.solution_value() for s in subset_indicators], dtype=bool)
		min_cost = solver.Objective().Value()
		return soln, min_cost

def wset_cover(subsets: csc_matrix, weights: ArrayLike, method: str = "RR", **kwargs):
	"""Computes an approximate solution to the weighted set cover problem using a supplied method."""
	assert isinstance(method, str) and method.lower() in ["rr", "greedy", "ilp", "sat"], f"Invalid method '{str(method)}' supplied; must be one of 'LP', 'greedy', or 'SAT'"
	method = method.upper()
	if method == "RR":
		return wset_cover_RR(subsets, weights, **kwargs)
	elif method == "GREEDY":
		return wset_cover_greedy(subsets, weights, **kwargs)
	elif method == "ILP": 
		return wset_cover_ILP(subsets, weights, **kwargs)
	else:
		return wset_cover_sat(subsets, weights, **kwargs)

# # RC2(wcnf)

# ## Solve 
# from pysat.examples.rc2 import RC2
# # with RC2(wcnf) as rc2: 
# #   for assignment in rc2.enumerate():
# #     print('model {0} has cost {1}'.format(assignment, rc2.cost))

# # TODO: iterate through k-subsets!
# for k in range(2, 8):
#   cover = create_cover(k)
#   formula = maxsat_wcnf(cover, mean_proj)
#   assignment = RC2(formula).compute()
#   sol_ind = np.flatnonzero(np.array(assignment) >= 0)
#   print(f"size: {len(sol_ind)}, weight: {np.sum(mean_proj[sol_ind])}")

# Greedy provides an H_k-approximation for the weight k set cover, where H_k = \sum\limits_{i=1}^k 1/i is the k-th harmonic number
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.1615&rep=rep1&type=pdf
# See also concise bounds w/ references: https://www.cs.ucr.edu/~neal/Young08SetCover.pdf
# Nice slides: http://cs.williams.edu/~shikha/teaching/spring20/cs256/lectures/Lecture31.pdf

# def wset_cover_greedy_naive(n, S, W):
# 	''' 
# 	Computes a set of indices I \in [m] whose subsets S[I] = { S_1, S_2, ..., S_k }
# 	yield an approximation to the minimal weighted set cover, i.e.

# 		S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
# 				 such that S_1 \cup ... \cup S_k covers [n]
	
# 	Parameters: 
# 		n: int := The number of points the subsets must cover 
# 		S: (n x J) sparsematrix := A sparse matrix whose non-zero elements indicate the subsets (one subset per column-wise)
# 		W: ndarray(J,1) := weights for each subset 

# 	Returns: 
# 		C: ndarray(k,1) := set of indices of which subsets form a minimal cover 
# 	'''
# 	assert issparse(S)
# 	assert S.shape[0] == n and S.shape[1] == len(W)
# 	J = S.shape[1]

# 	def covered(I):
# 		membership = np.zeros(n, dtype=bool)
# 		for j in I: membership[np.flatnonzero(S[:,j].A)] = True
# 		return(membership)

# 	C = []
# 	membership = covered(C)
# 	while not(np.all(membership)):
# 		not_covered = np.flatnonzero(np.logical_not(membership))
# 		cost_effectiveness = []
# 		for j in range(J):
# 			S_j = np.flatnonzero(S[:,j].A)
# 			size_uncovered = len(np.intersect1d(S_j, not_covered))
# 			cost_effectiveness.append(size_uncovered/W[j])
# 		C.append(np.argmax(cost_effectiveness))
# 		# print(C[-1])
# 		membership = covered(C)
	
# 	## Return the greedy cover
# 	return(np.array(C))
