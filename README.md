`geomcover` is a Python package built to simplify constructing geometric set covers over a variety of geometric objects, 
such as metric graphs and manifolds. Such covers can be used for simplifying surfaces, constructing fiber bundles, 
or summarizing point cloud data.

The package currently includes efficient implementations of approximation algorithms for [the set cover problem](https://en.wikipedia.org/wiki/Set_cover_problem) (SCP), functions for constructing tangent bundles from range spaces, and tools for computing statistics on them (e.g. average alignment of tangent vectors). The package also exports useful auxiliary algorithms often used for preprocessing, such as [principle component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis), [classical multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling), and [mean shift](https://en.wikipedia.org/wiki/Mean_shift).  

Below are the SCP solvers implemented, along with their relative scalabilities, approximation qualities, and dependencies:

| Algorithm                   | Scalability   | Approximation Quality | Required Dependencies         | Solvers         |
|-----------------------------|---------------|-----------------------|-------------------------------|-----------------|
| Greedy                      | Very large $n$| Medium                | Numpy                         | Direct          |
| Randomized Rounding (LP)    | Large $n$     | High                  | Numpy / SciPy                 | `linprog`       |
| ILP Method                  | Medium $n$    | Optimal[^1]              | Numpy / SciPy (or `ortools`)  | `milp` (or MPSolver) |
| SAT Solver                  | Small $n$     | Optimal[^2]               | pysat                         |  RC2                   |

[^1]: The integer-linear program (ILP) by default uses SciPy's port of the the HiGHs solver, which typically finds the global optimum of moderately challenging problems; alternatively, any solver supported by OR-Tools [MPSolver](https://developers.google.com/optimization/lp/mpsolver) can be used if the `ortools` package is installed.  

[^2]: The SAT solver used here is a weighted MaxSAT solver, which starts with an approximation solution and then incrementally exhausts the solution space until the optimal solution. While it is the slowest method, the starting approximation has the tightest approximation factor of all the methods listed. 

## Installation 

The package is a pure Python package, which for now needs to be built from source, e.g.: 

```bash
python -m pip install git+https://github.com/peekxc/geomcover.git
```
A PYPI distribution is planned. 

## Usage and Docs

For some example usage, see the `notebooks` directory, or examples in the [API docs](https://peekxc.github.io/geomcover/). 

