import numpy as np
from itertools import islice, cycle, chain, combinations
from more_itertools import pairwise
from typing import Iterable, Generator
from scipy.sparse import coo_array


def sliding_window(S: Iterable, n: int = 2) -> Generator:
  """Generates a sliding window of width n from the iterable.
  
  In other words, this function maps a m-length sequence 's' to a generator of tuples: 
    
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..., (s[m-n],s[m-n+1],...,sm)              

  """
  it = iter(S)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for elem in it:
    result = result[1:] + (elem,)
    yield result

def cycle_window(S: Iterable, offset: int = 1, w: int = 2):
  return sliding_window(islice(cycle(S), len(S)+offset), w)

def complete_graph(n: int):
  """Creates the complete graph from the sequence [0,1,...,n-1]""" 
  G = np.ones(shape=(n,n), dtype=bool)
  np.fill_diagonal(G, 0)
  return coo_array(G)

def cycle_graph(n: int, k: int = 2):
  """Creates a cycle graph from the sequence [0,1,...,n-1], connecting all k-adjacent pairs (cyclically)"""
  S = np.fromiter(cycle_window(range(n), w=k, offset=k), dtype=(np.int32, k))
  E = np.fromiter(chain(*[combinations(s, 2) for s in S]), dtype=(np.int32, 2))
  E = np.unique(E, axis=0)
  # E = np.fromiter(chain(*[S[:,[i,j]] for i,j in pairwise(range(k))]), dtype=(np.int32, 2))
  G = coo_array((np.ones(len(E), dtype=bool), (E[:,0], E[:,1])), shape=(n,n))
  G = G + G.T
  return G

def path_graph(n: int, k: int = 2):
  """Creates a path graph from the sequence [0,1,...,n-1], connecting all k-adjacent pairs""" 
  S = np.fromiter(sliding_window(range(n), k), dtype=(np.int32, k))
  E = np.fromiter(chain(*[combinations(s, 2) for s in S]), dtype=(np.int32, 2))
  E = np.unique(E, axis=0)
  # E = np.fromiter(chain(*[S[:,[i,j]] for i,j in range(k)]), dtype=(np.int32, 2))
  G = coo_array((np.ones(len(E), dtype=bool), (E[:,0], E[:,1])), shape=(n,n))
  G = G + G.T
  return G
