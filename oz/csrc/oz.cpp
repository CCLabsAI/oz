#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "game.h"
#include "games/flipguess.h"

auto sigmoid_add(at::Tensor x, at::Tensor y) -> at::Tensor {
  return at::sigmoid(x + y);
}

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
#define XQUOTE(x) #x
#define QUOTE(x) XQUOTE(x)
  m.attr("__version__") = py::str(QUOTE(VERSION_INFO));
#else
  m.attr("__version__") = py::none();
#endif

  m.def("add_sigmoid", sigmoid_add);

  py::enum_<oz::player_t>(m, "Player")
      .value("Chance", oz::player_t::Chance)
      .value("P1", oz::player_t::P1)
      .value("P2", oz::player_t::P2);

  py::class_<oz::action_t>(m, "Action");

  py::class_<oz::infoset_t>(m, "Infoset")
      .def_property_readonly("actions", &oz::infoset_t::actions)
      .def("__str__", &oz::infoset_t::str)
      ;

  py::class_<oz::game_t>(m, "Game")
      .def("act", &oz::game_t::act)
      .def("infoset", &oz::game_t::infoset)
      .def("player", &oz::game_t::player)
      .def("is_terminal", &oz::game_t::is_terminal)
      .def("utility", &oz::game_t::utility)
      ;

  py::class_<oz::flipguess_t, oz::game_t>(m, "FlipGuess")
      .def("heads", &oz::flipguess_t::heads);

  m.def("make_flipguess", []() {
    return std::unique_ptr<oz::game_t>(new oz::flipguess_t);
  });
}
