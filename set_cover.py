from array import array
import numpy as np
from typing import *
from numpy.typing import ArrayLike
from scipy.sparse import issparse, csc_matrix
from scipy.optimize import linprog
# from test_set_cover import S


def wset_cover_LP(subsets: ArrayLike, weights: ArrayLike, maxiter: int = "default"):
	''' 
	Computes an approximate solution to the weighted set cover problem 
	
	Args:
		subsets: (n x J) sparse matrix of J subsets whose union forms a cover over n points  
		weights: (J)-length array of weights associated with each subset 
		maxiter: number of iterations to repeat the random sampling process. See details.  
	
	Returns: 
		Returns a tuple (soln, cost) where soln is a boolean vector indicating which subsets are included in the optimal solution 
		and cost is the minimized cost of that solution. 
	'''
	assert issparse(subsets), "cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	W = weights.reshape((1, len(weights))) # ensure W is a column vector
	
	# linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point', callback=None, options=None, x0=None)
	# Since A_ub @ x <= b_ub, negate A_ub to enable A_ub @ x >= 1
	soln = linprog(c=W, b_ub=np.repeat(-1.0, subsets.shape[0]), A_ub=-subsets, bounds=(0.0, 1.0))
	assert soln.success

	## Change maxiter to default solution
	if maxiter == "default":
		maxiter = 2*int(np.ceil(np.log(subsets.shape[0])))

	## Sample subsets elementwise with probability 'p_i' until achieving a valid cover.
	## Repeat the sampling process 'maxiter' times and choose the best one
	p = soln.x
	best_cost, best_soln = np.inf, np.repeat(True, len(p))
	for _ in range(maxiter):
		z = np.random.random_sample(len(p)) <= p
		while np.any((subsets @ z) < 1.0):
			z = np.logical_or(z, np.random.random_sample(len(p)) <= p)
		cost = W @ z
		if cost < best_cost:
			best_cost, best_soln = cost, z
	return(np.flatnonzero(best_soln), best_cost.item())

# Adapted from: http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
# Also see: https://courses.engr.illinois.edu/cs598csc/sp2011/Lectures/lecture_4.pdf
def wset_cover_greedy(subsets: csc_matrix, weights: ArrayLike):
	"""Find a family of subsets that covers the universal set"""
	assert issparse(subsets), "cover must be sparse matrix"
	assert len(weights) == subsets.shape[1], "Number of weights must match number of subsets"
	S, W = subsets, weights
	n, J = S.shape
	elements, sets, point_cover, set_cover = set(range(n)), set(range(J)), set(), array('I')
	slice_col = lambda j: S.indices[S.indptr[j]:S.indptr[j+1]] # provides efficient cloumn slicing

	# Flip weights to not run into divide by 0 errors
	if np.any(W == 0):
		W[W == 0] = 1/np.finfo(float).resolution
	W = 1.0/W

	# Greedily add the subsets with the most uncovered points
	while point_cover != elements:
		I = max(sets, key=lambda j: (1.0/W[j])*len(set(slice_col(j)) - point_cover) )
		set_cover.append(I)
		point_cover |= set(slice_col(I))
		sets -= set(set_cover)
	return((np.sort(set_cover), np.sum((1.0/W)[np.array(set_cover)])))

# Greedy provides an H_k-approximation for the weight k set cover, where H_k = \sum\limits_{i=1}^k 1/i is the k-th harmonic number
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.1615&rep=rep1&type=pdf
# See also concise bounds w/ references: https://www.cs.ucr.edu/~neal/Young08SetCover.pdf
# Nice slides: http://cs.williams.edu/~shikha/teaching/spring20/cs256/lectures/Lecture31.pdf

# def wset_cover_greedy(subsets: csc_matrix, weights: ArrayLike):
# 	S, W = subsets, weights
# 	n, J = S.shape
# 	#point_cover, set_cover = array('I'), array('I')
# 	point_cover = np.repeat(False, n)
# 	slice_col = lambda j: S.indices[S.indptr[j]:S.indptr[j+1]]
# 	while np.any(~point_cover):
		
# 		#candidate_sets = np.setdiff1d(range(J), set_cover)
# 		#cover_cost = [w*len(np.setdiff1d(slice_col(j), point_cover)) for j, w in zip(candidate_sets, W[candidate_sets])]
# 		#set_cover.append(candidate_sets[np.argmin(cover_cost)])
# 		#point_cover.extend(np.setdiff1d(slice_col(set_cover[-1]), point_cover))


def wset_cover_greedy_naive(n, S, W):
	''' 
	Computes a set of indices I \in [m] whose subsets S[I] = { S_1, S_2, ..., S_k }
	yield an approximation to the minimal weighted set cover, i.e.

		S* = argmin_{I \subseteq [m]} \sum_{i \in I} W[i]
				 such that S_1 \cup ... \cup S_k covers [n]
	
	Parameters: 
		n: int := The number of points the subsets must cover 
		S: (n x J) sparsematrix := A sparse matrix whose non-zero elements indicate the subsets (one subset per column-wise)
		W: ndarray(J,1) := weights for each subset 

	Returns: 
		C: ndarray(k,1) := set of indices of which subsets form a minimal cover 
	'''
	assert issparse(S)
	assert S.shape[0] == n and S.shape[1] == len(W)
	J = S.shape[1]

	def covered(I):
		membership = np.zeros(n, dtype=bool)
		for j in I: membership[np.flatnonzero(S[:,j].A)] = True
		return(membership)

	C = []
	membership = covered(C)
	while not(np.all(membership)):
		not_covered = np.flatnonzero(np.logical_not(membership))
		cost_effectiveness = []
		for j in range(J):
			S_j = np.flatnonzero(S[:,j].A)
			size_uncovered = len(np.intersect1d(S_j, not_covered))
			cost_effectiveness.append(size_uncovered/W[j])
		C.append(np.argmax(cost_effectiveness))
		# print(C[-1])
		membership = covered(C)
	
	## Return the greedy cover
	return(np.array(C))
