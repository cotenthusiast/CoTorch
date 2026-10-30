#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cotorch/matmul.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.def("matmul", &matmul);
}