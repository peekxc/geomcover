# https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
# The format of all of these 80 data files is:
# number of rows (m), number of columns (n)
# the cost of each column c(j),j=1,...,n
# for each row i (i=1,...,m): the number of columns which cover
# row i followed by a list of the columns which cover row i

# np.loadtxt("https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/CCNFP10g1b.txt")
import numpy as np
from scipy.sparse import csc_array

from geomcover import load_set_cover
from geomcover import set_cover_ilp, set_cover_rr, set_cover_greedy, set_cover_sat
from geomcover.cover import valid_cover, coverage, to_canonical
from geomcover.io import OR_TEST_FILES

benchmarks = {}
for test_file in OR_TEST_FILES:
  A, weights = load_set_cover(test_file)
  benchmarks[test_file] = {
    'greedy' : wset_cover_greedy(A, weights),
    'RR' : wset_cover_RR(A, weights),
    'ILP' : wset_cover_ILP(A, weights),
    'SAT' : wset_cover_sat(A, weights)
  }
  print(test_file)


A, weights = load_set_cover(test_file)

import timeit
timeit.timeit(lambda: wset_cover_greedy(A, weights), number=150)
# timeit.timeit(lambda: wset_cover_greedy2(A, weights), number=150)

assert valid_cover(A, np.flatnonzero(soln))

## accidents = benchmark at 4-45 seconds, solution from 181-245
freq_item_sets = ["mushroom.dat", "accidents.dat"]
with open("/Users/mpiekenbrock/geomcover/notebooks/accidents.dat", 'r') as f:
  x = list(f.readlines())

# https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
# http://dimacs.rutgers.edu/~graham/pubs/papers/ckw.pdf
sets = [list(map(int, l.replace('\n', '').strip().split(' '))) for l in x]
col_ind = np.concatenate([np.repeat(i, len(s)) for i,s in enumerate(sets)])
row_ind = np.concatenate(sets) - 1
assert np.all(np.unique(row_ind) == np.arange(np.max(row_ind)+1))

subsets = csc_array((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(np.max(row_ind)+1, len(sets)))
subsets = to_canonical(subsets, "csc")

weights = np.ones(subsets.shape[1])
soln, cost = set_cover_greedy(subsets, weights)  # 3.2s, 181 cost
soln, cost = set_cover_rr(subsets, weights)      # 10s, 158 cost
soln, cost = set_cover_ilp(subsets, weights)     # 13.5s, 158 cost
soln, cost = set_cover_sat(subsets, weights)     # 80.7s, 158 cost

np.min(coverage(subsets))

# BAsed on: https://algnotes.info/on/obliv/lagrangian/set-cover-fractional/
## simply doesn't work!
# C = 1
# c = np.ones(A.shape[1])#weights
# eps = 0.5
# # xe = coverage(A)
# B = to_canonical(A, "coo")
# xs = np.zeros(A.shape[1])
# ii = 0
# while np.min(coverage(B, np.flatnonzero(xs))) < 5:
#   xe = np.array([np.sum(xs[B.row[B.col == j]]) for j in range(B.shape[1])]) # coverage per element?
#   opt_s = np.argmax([np.sum(np.power(1-eps, xe[B.row[B.col == j]])) / c[j] for j in range(B.shape[1])])
#   xs[opt_s] -= 1
#   print(opt_s)
#   ii += 1
#   if ii == 15:
#     break

coverage(A, xs)

A = np.ones(shape=(10000,10000))
timeit.timeit(lambda: eval("partial(lambda j, A: A[:,j], A=A)(0)"), number=1500)
timeit.timeit(lambda: eval("(lambda j: A[:,j])(0)"), number=1500)



subsets.astype(int).todense()[0,:]


from set_cover.wset_cover import _cover, wset_cover_ILP
dir(_cover)
## TODO: fix
ind = _cover.greedy_set_cover(A.indices, A.indptr, w, A.shape[0])

wset_cover_greedy(A, w, info=True)

np.flip(np.sort(ind[:82]))
valid_cover(A, ind)

# valid_cover(A, np.flatnonzero(soln))

len(np.flatnonzero(soln))

from ortools.linear_solver import pywraplp
from set_cover.loaders import to_canonical


solver = pywraplp.Solver.CreateSolver("SCIP") # mip solver with the SCIP backend.
assert solver is not None
subset_indicators = [solver.IntVar(0,1,"") for i in range(A.shape[1])]
min_weight_obj = solver.Sum([s*w for s, w in zip(subset_indicators, weights)])
B = to_canonical(A, form="csr")
for z in np.split(B.indices, B.indptr)[1:-1]:
  solver.Add(solver.Sum([subset_indicators[zi] for zi in z]) >= 1)
solver.Minimize(min_weight_obj)
status = solver.Solve()
soln = np.array([s.solution_value() for s in subset_indicators], dtype=bool)
min_cost = solver.Objective().Value()
assert valid_cover(A, np.flatnonzero(soln))


# z = np.split(A.indices, A.indptr)[1:-1][0]
constraint_sat = np.array([
  np.sum([subset_indicators[zi].solution_value() for zi in z])
  for z in np.split(A.indices, A.indptr)[1:-1]
])
covered(A, np.flatnonzero(soln))

np.any(np.array(constraint_sat) == 0)

# from set_cover.covers import covered
# covered(A, np.flatnonzero(soln))



np.split(A.indices, A.indptr[[0,1,2,-1,-1]])[1:-1]

# def subset_cols(A: csc_array, col_ind: np.ndarray):
#   B = A.tocoo()
#   covered = np.zeros(A.shape[0], dtype=bool)
#   np.add.at(covered, B.row[np.isin(B.col, col_ind)], 1)
#   return np.all(covered)





from set_cover.covers import valid_cover
valid_cover(A, np.flatnonzero(soln))
A.indices

A.indices

solver.IntVar(0, 1, "")
solver.Sum()

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



##
import random
num_sets = 100000
set_size = 40
elements = list(range(500))
U = set(elements)
R = U
S = []
for i in range(num_sets):
    random.shuffle(elements)
    S.append(set(elements[:set_size]))
w = [random.randint(1,100) for i in range(100)]



def greedy_set_cover_simple(U, S, w):
  assert len(w) == len(S)
  C = []
  costs = []
  def findMin(S, R):
    minCost = 99999.0
    minElement = -1
    for i, s in enumerate(S):
      try:
        cost = w[i]/(len(s.intersection(R)))
        if cost < minCost:
            minCost = cost
            minElement = i
      except:
        # Division by zero, ignore
        pass
    return S[minElement], w[minElement]
  R = U
  while len(R) != 0:
    S_i, cost = findMin(S, R)
    C.append(S_i)
    R = R.difference(S_i)
    costs.append(cost)
  return C, costs

## 4.8 seconds
w = np.random.choice(range(1, 101), size=len(S)) #np.array(w)

timeit.timeit(lambda: greedy_set_cover_simple(U, S, w), number=15)
soln, cost =

from geomcover.cover import wset_cover_greedy, sets_to_csc
A = sets_to_csc(S)      # 0.6s
wset_cover_greedy(A, w) # 0.4s
