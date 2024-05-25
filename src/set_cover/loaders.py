import numpy as np
import re
from array import array
from scipy.sparse import coo_array, csc_matrix, csc_array

def clean_sp_mat(A, form: str = "csc", diag: bool = False, symmetrize: bool = False):
  assert isinstance(form, str) and form.lower() in ['csc', 'csr', 'lil', 'dok', 'coo'], f"Invalid form '{form}'; must be a format supported by SciPy."
  A = getattr(A, 'to' + form)()
  if symmetrize: 
    A += A.T
  if diag: 
    A.setdiag(True)
  if form in ['csr', 'csc']:
    A.eliminate_zeros()
    ## https://stackoverflow.com/questions/28428063/what-does-csr-matrix-sort-indices-do
    A.has_sorted_indices = False
    A.sort_indices()
    assert A.has_canonical_format, "Failed to clean sparse matrix"
  return A

## To load data sets from: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
def load_set_cover_instance(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  num_elements, num_sets = map(int, lines[0].split())
  data = ''.join(lines[1:]).replace("\n ", '')
  data = np.array([float(i) for i in re.split(r'\s+', data) if len(i) > 0])
  set_weights = data[:num_sets]
  data = data[num_sets:]
  cc = 0
  R,C = array('I'), array('I')
  for j in range(num_elements):
    nj = int(data[cc])
    row_j = data[(cc+1):(cc+nj+1)]
    cc += (nj + 1)
    C.extend(row_j.astype(int) - 1)
    R.extend(j * np.ones(nj, dtype=int))
  A = coo_array((np.ones(len(R)), (R, C)), shape=(num_elements, num_sets), dtype=bool)
  return clean_sp_mat(A), set_weights