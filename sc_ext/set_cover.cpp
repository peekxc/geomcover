#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

int testing(){
  return 0; 
}

PYBIND11_MODULE(set_cover, m) {
  m.def("testing", &testing);
};