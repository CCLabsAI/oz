#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "game.h"
#include "oss.h"

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

  py::class_<oz::history_t>(m, "History")
      .def("act", &oz::history_t::act)
      .def("infoset", &oz::history_t::infoset)
      .def_property_readonly("player", &oz::history_t::player)
      .def("is_terminal", &oz::history_t::is_terminal)
      .def("utility", &oz::history_t::utility)
      .def("__copy__", [](const oz::history_t &h){ return oz::history_t(h); });

  auto py_OSS =
  py::class_<oz::oss_t::search_t>(m, "Search")
      .def(py::init<oz::history_t, oz::player_t>())
      .def_property_readonly("state", &oz::oss_t::search_t::state)
      .def("infoset", &oz::oss_t::search_t::infoset);

  py::enum_<oz::oss_t::search_t::state_t>(py_OSS, "State")
      .value("SELECT", oz::oss_t::search_t::state_t::SELECT)
      .value("CREATE", oz::oss_t::search_t::state_t::CREATE)
      .value("PLAYOUT", oz::oss_t::search_t::state_t::PLAYOUT)
      .value("BACKPROP", oz::oss_t::search_t::state_t::BACKPROP)
      .value("FINISHED", oz::oss_t::search_t::state_t::FINISHED);

  m.def("make_flipguess", []() {
    return std::unique_ptr<oz::game_t>(new oz::flipguess_t);
  });

  m.def("make_flipguess_history", []() {
    return oz::make_history<oz::flipguess_t>();
  });

  m.def("make_kuhn", []() {
    return std::unique_ptr<oz::game_t>(new oz::kuhn_poker_t);
  });
}
