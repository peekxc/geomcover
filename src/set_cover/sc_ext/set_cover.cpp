#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

// S, W = subsets, weights
// n, J = S.shape
// elements, sets, point_cover, set_cover = set(range(n)), set(range(J)), set(), array('I')
// slice_col = lambda j: S.indices[S.indptr[j]:S.indptr[j+1]] # provides efficient cloumn slicing

// ## Make infinite costs finite, but very large
// if np.any(W == np.inf):
// 	W[W == np.inf] = 1.0/np.finfo(float).resolution

// # Greedily add the subsets with the most uncovered points
// while point_cover != elements:
// 	#I = min(sets, key=lambda j: W[j]/len(set(slice_col(j)) - point_cover) ) # adding RHS new elements to cover incurs weighted cost of w/|RHS|
// 	I = min(sets, key=lambda j: np.inf if (p := len(set(slice_col(j)) - point_cover)) == 0.0 else W[j]/p)
// 	set_cover.append(I)
// 	point_cover |= set(slice_col(I))
// 	sets -= set(set_cover)
// assignment = np.zeros(J, dtype=bool)
// assignment[set_cover] = True
// return((assignment, np.sum(weights[assignment])))

// Counter to avoid storing the set difference 
struct Counter {
  struct value_type { template<typename T> value_type(const T&) { } };
  void push_back(const value_type&) noexcept { ++count; }
  size_t count = 0;
};

template< typename Iter1, typename Iter2 >
size_t setdiff_size(Iter1 b1, const Iter1 e1, Iter2 b2, const Iter2 e2) noexcept {
  Counter c;
  std::set_difference(b1, e1, b2, e2, std::back_inserter(c));
  return c.count;
}

using std::begin;
using std::end; 
using std::vector; 
py::array_t< int > greedy_set_cover(py::array_t< int >& indices, py::array_t< int >& indptr, py::array_t<double>& weights, const size_t n){
  const size_t J = weights.size();

  auto I = indptr.unchecked<1>();
  auto ind = vector< int >(indices.size()); 
  for (size_t i = 0; i < ind.size(); ++i){
    ind[i] = I[i];
  }
  auto IP = indptr.unchecked<1>();
  auto W = weights.unchecked<1>();

  // Point indices in the current cover (initialized to empty set)
  auto pci = vector< int >();
  pci.reserve(n);

  // Candidate sets to choose from
  auto cand_sets = vector< int >(J);
  std::iota(cand_sets.begin(), cand_sets.end(), 0);

  // Priorities on the set of candidate sets
  auto set_imports = vector< double >();
  set_imports.reserve(n);
  
  // Actual indices of the sets making up the solution
  auto soln = vector< int >();

  size_t cc = 0; 
  while(pci.size() < n || cc < n){
    
    // Main computational set: get the sizes of the all the set differences
    set_imports.clear();
    for (size_t ji = 0; ji < cand_sets.size(); ++ji){
      size_t j = cand_sets[ji];
      auto jb = ind.begin()+IP[j];
      auto je = ind.begin()+IP[j+1];
      const size_t I_sz = setdiff_size(jb, je, pci.begin(), pci.end());
      set_imports.push_back(I_sz == 0 ? std::numeric_limits<double>::infinity() : W[j]/I_sz);
    }
    
    // Greedy step
    auto min_it = std::min_element(set_imports.begin(), set_imports.end());
    auto min_ind = std::distance(set_imports.begin(), min_it);
    const size_t best_j = cand_sets[min_ind];
    cand_sets.erase(cand_sets.begin()+best_j); // remove the set from future consideration

    // Union the best set into the point cover
    auto jb = ind.begin()+IP[best_j];
    auto je = ind.begin()+IP[best_j+1];
    std::set_difference(jb, je, pci.begin(), pci.end(), std::back_inserter(pci));
    if (set_imports[best_j] == std::numeric_limits< double >::max()){
      // set difference empty
      soln.push_back(best_j);
    } else {
      const size_t j_sz = static_cast< size_t >(set_imports[best_j]*W[best_j]);
      std::inplace_merge( pci.begin(), pci.begin() + j_sz, pci.end()); 
      soln.push_back(best_j);
    }

    cc++; 
  }

  py::array_t< int > soln_np(soln.size(), soln.data());
  return(soln_np);
}

PYBIND11_MODULE(set_cover, m) {
  m.def("greedy_set_cover", &greedy_set_cover);
};