#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "game.h"
#include "games/flipguess.h"
#include "games/kuhn.h"

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

  auto py_Player =
    py::enum_<oz::player_t>(m, "Player")
      .value("Chance", oz::player_t::Chance)
      .value("P1", oz::player_t::P1)
      .value("P2", oz::player_t::P2);

  py::class_<oz::action_t>(m, "Action");

  py::class_<oz::infoset_t>(m, "Infoset")
      .def_property_readonly("actions", &oz::infoset_t::actions)
      .def("__str__", &oz::infoset_t::str);

  auto py_Game =
    py::class_<oz::game_t>(m, "Game")
      .def("act", &oz::game_t::act)
      .def("infoset", &oz::game_t::infoset)
      .def_property_readonly("player", &oz::game_t::player)
      .def("is_terminal", &oz::game_t::is_terminal)
      .def("utility", &oz::game_t::utility)
      .def("__copy__", [](const oz::game_t &game){ return game.clone(); });

  py_Game.attr("Player") = py_Player;

  py::class_<oz::flipguess_t, oz::game_t>(m, "FlipGuess")
      .def("heads", &oz::flipguess_t::heads);

  py::class_<oz::kuhn_poker_t, oz::game_t>(m, "KuhnPoker");

  m.def("make_flipguess", []() {
    return std::unique_ptr<oz::game_t>(new oz::flipguess_t);
  });

  m.def("make_kuhn", []() {
    return std::unique_ptr<oz::game_t>(new oz::kuhn_poker_t);
  });
}
