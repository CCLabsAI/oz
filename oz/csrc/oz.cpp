#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#endif

}
