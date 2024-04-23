# https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
# The format of all of these 80 data files is:
# number of rows (m), number of columns (n)
# the cost of each column c(j),j=1,...,n
# for each row i (i=1,...,m): the number of columns which cover
# row i followed by a list of the columns which cover row i

# np.loadtxt("https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/CCNFP10g1b.txt")
import numpy as np 
from scipy.sparse import csc_array, coo_array

from set_cover import load_set_cover_instance
from set_cover import wset_cover, wset_cover_LP, wset_cover_greedy, wset_cover_sat
from set_cover.covers import valid_cover

A, w = load_set_cover_instance("/Users/mpiekenbrock/set_cover/notebooks/scp41.txt")

soln, cost = wset_cover_LP(A, w)
soln, cost = wset_cover_greedy(A, w)
soln, cost = wset_cover_sat(A, w)

from set_cover.wset_cover import _cover
dir(_cover)
## TODO: fix
ind = _cover.greedy_set_cover(A.indices, A.indptr, w, A.shape[0])

wset_cover_greedy(A, w, info=True)

np.flip(np.sort(ind[:82]))
valid_cover(A, ind)

# valid_cover(A, np.flatnonzero(soln))

len(np.flatnonzero(soln))

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

solver = pywraplp.Solver.CreateSolver("SCIP") # mip solver with the SCIP backend.
assert solver is not None

## Define variables 
x = {}
for j in range(data["num_vars"]):
  x[j] = solver.IntVar(0, infinity, "x[%i]" % j)

B.indices

constraint = solver.RowConstraint(0, 1, "")



z1 = solver.BoolVar("z1")
solver.Add(x + 7 * y <= 17.5)

solver.Minimize()
status = solver.Solve()


A = csc_array(np.loadtxt("notebooks/camera_stadium.txt").astype(bool)).sorted_indices()


set_weights = np.ones(A.shape[1])

soln, wght = wset_cover_greedy(A, weights=set_weights)
np.flatnonzero(soln) + 1

soln, wght = wset_cover_LP(A, weights=set_weights)
np.flatnonzero(soln) + 1

soln, wght = wset_cover_sat(A, weights=set_weights)
np.flatnonzero(soln) + 1

# wset_cover_sat(A, weights=np.ones(A.shape[1]))

# from pysat.examples.rc2 import RC2
# from pysat.formula import WCNF
# # subsets = np.array([[0,1,2,3,4,5], [4,5,7,8], [0,3,]])
# # A = csc_array(np.array([[1,0,1,0,0],[1,0,0,1,1],[1,1,0,0,1],[0,0,1,1,0],[0,0,0,1,1],[0,1,0,0,1]]), dtype=bool)
# n, J = A.shape
# B = A.T.tocsc().sorted_indices()
# wcnf = WCNF()

# ## Covering constraint: all elements must be covered
# N = [z for z in np.split(B.indices+1, B.indptr)[1:-1]]
# wcnf.extend(N, weights=None)

# ## Soft constraint: encourage less subsets by accumulating negative variables 
# wcnf.extend(-(np.arange(J)+1)[:,np.newaxis], weights=list(np.ones(J)))

# # ## Now add negative weights for each cover subset 
# # for j in range(J):
# #   wcnf.append([-(j+1)], weight=1)

# wcnf = WCNF(from_string=wcnf.to_dimacs())


# solver = RC2(wcnf)
# solver.compute()

# valid_cover(A, [0,1,3])

# #list(np.ones(J, dtype=int))


# # for x, w in zip(np.split(subsets.indices+1, subsets.indptr)[1:-1], list(np.ones(J, dtype=int))):
# #   wcnf.append(x, weight=-w)
# # wcnf.extend(np.split(subsets.indices+1, subsets.indptr)[1:-1], weights=list(np.ones(J, dtype=int)))
# # wcnf.extend(list(-((np.arange(J)+1))[:,np.newaxis]), weights=None)
# wcnf.extend([[i] for i in (np.arange(n)+1)], weights=None)
# print(wcnf.to_dimacs())

# wcnf = WCNF(from_string=wcnf.to_dimacs())
# #wcnf.hard
# #wcnf.soft
# solver = RC2(wcnf)
# solver.compute()

# ind = np.flatnonzero(np.array(solver.compute()) > 0)
# valid_cover(subsets, ind)

# # 15 variables, 23 clauses,



# print(wcnf.to_dimacs())


# for subset in np.split(subsets.indices, subsets.indptr)[1:-1]:
#   wcnf.append(list((subset+1).astype(int)), weight=None)

# for j, w in enumerate(np.ones(subsets.shape[0])): 
#   wcnf.append([-int(j+1)], weight=int(w))

# print(wcnf.to_dimacs())

# from tempfile import NamedTemporaryFile
# tmp = NamedTemporaryFile()
# wcnf.to_file(tmp.name)
# wcnf = WCNF(from_file=tmp.name)
# tmp.close()

# print(wcnf.to_dimacs())

# solver = RC2(wcnf)

# from pysat.formula import WCNF
# cnf = WCNF()
# cnf.append([-1, 2])
# cnf.append([1], weight=10.0)
# cnf.append([-2], weight=20.0)
# RC2(cnf)