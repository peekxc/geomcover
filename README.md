`geomcover` is a Python package built to simplify constructing geometric set covers over a variety of geometric objects, 
such as metric graphs and manifolds. Such covers can be used for simplifying surfaces, constructing fiber bundles, 
or summarizing point cloud data. The package includes functions for approximating the [the set cover problem](https://en.wikipedia.org/wiki/Set_cover_problem) (SCP), for constructing tangent bundles from _range spaces_, and for computing statistics on tangent spaces (e.g. average alignment of tangent vectors). 

## Set Cover solvers

Though primarily intended to work with geometric inputs (e.g. point cloud data), `geomcover` can also be used as an interface to 
aa variety of solvers for the [weighted] set cover problem (SCP), which are summarized below:

| Algorithm                   | Scalability    | Approx. Quality | Dependencies         | Solvers         |
|-----------------------------|----------------|-----------------------|-------------------------------|-----------------|
| Greedy                      | Very large $n$ | Medium                | Numpy                         | Direct          |
| Randomized Rounding (LP)    | Large $n$      | High                  | SciPy                 | `linprog`       |
| ILP Method                  | Medium $n$     | Optimal[^1]              | SciPy (or `ortools`)  | `milp` (or MPSolver) |
| SAT Solver                  | Small $n$      | Optimal[^2]               | pysat                         |  RC2                   |

Usage is as follows: 
```python
from geomcover.cover import set_cover

cover, cover_weight = set_cover(subsets, method = "ILP", ...)
```

For more example, see the [API docs](https://peekxc.github.io/geomcover/). The package also exports useful auxiliary algorithms often used for preprocessing, such as [principle component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), [classical multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), and [mean shift](https://en.wikipedia.org/wiki/Mean_shift).  

## Installation 

The package is a pure Python package, which for now needs to be built from source, e.g.: 

```bash
python -m pip install git+https://github.com/peekxc/geomcover.git
```
A PYPI distribution is planned. 

## Usage and Docs

For some example usage, see the `notebooks` directory, or examples in the [API docs](https://peekxc.github.io/geomcover/). 


[^1]: The integer-linear program (ILP) by default uses SciPy's port of the the HiGHs solver, which typically finds the global optimum of moderately challenging problems; alternatively, any solver supported by OR-Tools [MPSolver](https://developers.google.com/optimization/lp/mpsolver) can be used if the `ortools` package is installed.  

[^2]: The SAT solver used here is a weighted MaxSAT solver, which starts with an approximation solution and then incrementally exhausts the solution space until the optimal solution. While it is the slowest method, the starting approximation has the tightest approximation factor of all the methods listed. 