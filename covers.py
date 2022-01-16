import numpy as np
from tallem.dimred import neighborhood_graph, neighborhood_list, pca
from scipy.sparse import csr_matrix, csc_matrix, find

def neighborhood_cover(X, d, method=["tangent_proj"], ind = None, include_self=True, **kwargs):
  ''' 
  Constructs a weighted cover by constructing a neighborhood around every point and then 
  computing some predefined local functional on the points within each neighborhood

  Parameters: 
    X := (n x d) point cloud data in Euclidean space, or and (n x n) sparse adjacency matrix yielding a weighted neighborhood graph
    d := local dimension where the metric is approximately Euclidean
    method := choice of neighborhood criteria

  Returns: 
    cover := (n x J) csc_matrix 
    weights := J-length array of weights computed using 'method'
  '''
  G = neighborhood_graph(X, **kwargs) if ind is None else neighborhood_list(X[ind,:], X, **kwargs).astype(bool)
  if "k" in list(kwargs.keys()):
    G_dense = G.A
    G_dense += G_dense.T
    np.fill_diagonal(G_dense, include_self) # always include self
    G = csc_matrix(G_dense) 
  else:
    G_dense = G.A
    np.fill_diagonal(G_dense, True)
    G = csc_matrix(G_dense) 
  r,c,v = find(G)
  weights = np.zeros(X.shape[0])
  tangents = [None]*X.shape[0]
  for i, x in enumerate(X): 
    nn_idx = c[r == i] #np.append(np.flatnonzero(G[i,:].A), i)
    if len(nn_idx) < 2: 
      raise ValueError("bad cover")
    
    ## Get tangent space estimates at centered points
    centered_pts = X[nn_idx,:]-x
    _, T_y = pca(centered_pts, d=d, coords=False)
    tangents[i] = T_y # ambient x local

    ## Project all points onto tangent plane, then measure distance between projected points and original
    proj_coords = np.dot(centered_pts, T_y) # project points onto d-tangent plane
    proj_points = np.array([np.sum(p*T_y, axis=1) for p in proj_coords]) # orthogonal projection in D dimensions
    weights[i] = np.sum([np.sqrt(np.sum(diff**2)) for diff in (centered_pts - proj_points)]) # np.linalg.norm(centered_pts - proj_points)
  
  assert np.all(G.A == G.A.T)
  return(G, weights, tangents)
