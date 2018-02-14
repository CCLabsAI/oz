#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(oz, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#endif
}
