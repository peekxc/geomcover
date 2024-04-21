import os 
import importlib
import numpy as np
from array import array
from typing import *
from numpy.typing import ArrayLike
from scipy.sparse import issparse, csc_matrix
from scipy.optimize import linprog

def package_exists(package: str) -> bool: 
	pkg_spec = importlib.util.find_spec(package)
	return(pkg_spec is not None)

def ask_package_install(package: str):
	if not(package_exists(package)):
		raise RuntimeError(f"Module {package} not installed. To use this function, please install {package}.")

def wset_cover_LP(subsets: ArrayLike, weights: ArrayLike, maxiter: int = "default"):
	''' 
	Computes an approximate solution to the weighted set cover problem via Linear Programming (LP) sampling 
	
	Args:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points  
		weights: (J)-length array of weights associated with each subset 
		maxiter: number of iterations to repeat the sampling process. See details.  
	
	If not supplied, maxiter defaults to 2*log(n)

	Returns: 
		(s, c) := tuple where s is a boolean vector indicating which subsets are included in the optimal solution 
		and c is the minimized cost of that solution. 
	'''
	assert issparse(subsets), "cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	W = weights.reshape((1, len(weights))) # ensure W is a column vector
	if subsets.dtype != int:
		subsets = subsets.astype(int)

	# linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point', callback=None, options=None, x0=None)
	# Since A_ub @ x <= b_ub, negate A_ub to enable constraint A_ub @ x >= 1
	soln = linprog(c=W, b_ub=np.repeat(-1.0, subsets.shape[0]), A_ub=-subsets, bounds=(0.0, 1.0), options={"sparse": True})
	assert soln.success

	## Change maxiter to default solution
	if maxiter == "default":
		maxiter = 2*int(np.ceil(np.log(subsets.shape[0])))

	## Sample subsets elementwise with probability 'p_i' until achieving a valid cover.
	## Repeat the sampling process 'maxiter' times and choose the best one
	p = soln.x
	best_cost, assignment = np.inf, np.zeros(len(p), dtype=bool)
	for _ in range(maxiter):
		z = np.random.random_sample(len(p)) <= p
		while np.any((subsets @ z) < 1.0): # while any point is left uncovered
			z = np.logical_or(z, np.random.random_sample(len(p)) <= p)
		cost = np.dot(W, z)
		best_cost, assignment = (cost, z) if cost < best_cost else (best_cost, assignment)
	return(assignment, best_cost)

# Adapted from: http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
# Also see: https://courses.engr.illinois.edu/cs598csc/sp2011/Lectures/lecture_4.pdf
def wset_cover_greedy(subsets: csc_matrix, weights: ArrayLike):
	"""
	Computes a set of indices I whose subsets S[I] = { S_1, S_2, ..., S_k }
	yield an approximation to the minimal weighted set cover, i.e.

		S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
				 s.t. union(S_1, ..., S_k) covers [n]
	
	Parameters: 
		S: An (n x m) sparse matrix whose non-zero elements indicate subset membership (one subset per column)
		W: Weights for each subset 

	Returns: 
		(s, c) := tuple where s is a boolean vector indicating which subsets are included in the optimal solution 
		and c is the minimized cost of that solution. 
	"""
	assert issparse(subsets), "cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	S, W = subsets, weights
	n, J = S.shape
	elements, sets, point_cover, set_cover = set(range(n)), set(range(J)), set(), array('I')
	slice_col = lambda j: S.indices[S.indptr[j]:S.indptr[j+1]] # provides efficient cloumn slicing

	## Make infinite costs finite, but very large
	if np.any(W == np.inf):
		W[W == np.inf] = 1.0/np.finfo(float).resolution

	# Greedily add the subsets with the most uncovered points
	while point_cover != elements:
		#I = min(sets, key=lambda j: W[j]/len(set(slice_col(j)) - point_cover) ) # adding RHS new elements to cover incurs weighted cost of w/|RHS|
		I = min(sets, key=lambda j: np.inf if (p := len(set(slice_col(j)) - point_cover)) == 0.0 else W[j]/p)
		set_cover.append(I)
		point_cover |= set(slice_col(I))
		sets -= set(set_cover)
	assignment = np.zeros(J, dtype=bool)
	assignment[set_cover] = True
	return((assignment, np.sum(weights[assignment])))


def _maxsat_wcnf(subsets: csc_matrix, weights: ArrayLike):
	""" Produces a WMAX-SAT CNF formula"""
	from pysat.formula import WCNF
	from tempfile import TemporaryFile, NamedTemporaryFile
	assert subsets.shape[1] == len(weights)
	subsets = subsets.astype(bool) 
	wcnf = WCNF()
	for j in range(subsets.shape[1]): 
		wcnf.append(list(np.flatnonzero(subsets[:,j].A)+1), weight=None)
	for j, w in enumerate(weights): 
		wcnf.append([-int(j+1)], weight=w)
	
	## The types are not inferred from direct inputs, so have to write to file to get correct parsing
	tmp = NamedTemporaryFile()
	wcnf.to_file(tmp.name)
	wcnf = WCNF(from_file=tmp.name)
	tmp.close()
	return(wcnf)

def wset_cover_sat(subsets: csc_matrix, weights: ArrayLike, return_solver: bool = False, **kwargs):
	ask_package_install("pysat")
	from pysat.examples.rc2 import RC2
	wcnf = _maxsat_wcnf(subsets, weights)
	solver = RC2(wcnf, **kwargs)
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
