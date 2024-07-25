import numpy as np
import re
from array import array
from scipy.sparse import coo_array
from urllib.request import urlopen

from .csgraph import to_canonical

## J. E. Beasley's collection of test data sets for the OR-lib
## From: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html
OR_TEST_FILES = [
  'scp41', 'scp42', 'scp43', 'scp44', 'scp45', 'scp46', 'scp47', 'scp48', 'scp49', 'scp410',
  'scp51', 'scp52', 'scp53', 'scp54', 'scp55', 'scp56', 'scp57', 'scp58', 'scp59', 'scp510',
  'scp61', 'scp62', 'scp63', 'scp64', 'scp65',
  'scpa1', 'scpa2', 'scpa3', 'scpa4', 'scpa5',
  'scpb1', 'scpb2', 'scpb3', 'scpb4', 'scpb5',
  'scpc1', 'scpc2', 'scpc3', 'scpc4', 'scpc5',
  'scpd1', 'scpd2', 'scpd3', 'scpd4', 'scpd5',
  'scpe1', 'scpe2', 'scpe3', 'scpe4', 'scpe5'
]

## To load data sets from: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/
def load_set_cover_instance(test_set: str):
  assert test_set.lower() in OR_TEST_FILES, f"Unknown test data set '{test_set}'; must be a set cover instance"
  data = urlopen(f"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{test_set.lower()}.txt")
  lines = data.readlines()
  lines = [ln.decode() for ln in lines]
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
  return to_canonical(A, "coo"), set_weights