import numpy as np
import re
from array import array
from scipy.sparse import coo_array

def _clean_sp_mat(A):
	A = A.tocsc() if not hasattr(A, "indices") else A 
	A.eliminate_zeros()
	A.sort_indices()
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
    return _clean_sp_mat(A), set_weights