#include "cotorch/matmul.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.def("matmul", &matmul);
  m.def("benchmark_matmul", &benchmark_matmul);
}
