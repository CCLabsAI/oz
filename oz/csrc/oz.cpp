#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "game.h"
#include "best_response.h"
#include "oos.h"
#include "batch.h"
#include "encoder.h"
#include "py_sigma.h"
#include "games/flipguess.h"
#include "games/kuhn.h"
#include "games/leduk.h"

auto sigmoid_add(at::Tensor x, at::Tensor y) -> at::Tensor {
  return at::sigmoid(x + y);
}

namespace py = pybind11;

void bind_oz(py::module &m);

PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
#define XQUOTE(x) #x
#define QUOTE(x) XQUOTE(x)
  m.attr("__version__") = py::str(QUOTE(VERSION_INFO));
#else
  m.attr("__version__") = py::none();
#endif

  m.def("sigmoid_add", sigmoid_add);
  bind_oz(m);
}

void bind_oz(py::module &m) {
  using namespace oz;

  auto py_Player =
      py::enum_<player_t>(m, "Player")
          .value("Chance", player_t::Chance)
          .value("P1", player_t::P1)
          .value("P2", player_t::P2)
          .export_values();

  py::class_<action_t>(m, "Action")
      .def_property_readonly("index", &action_t::index)
      .def("__eq__", [](const action_t &self, const action_t &other) {
        return self.index() == other.index();
      })
      .def("__hash__", [](const action_t &self) {
        return py::hash(py::int_(self.index()));
      });

  py::class_<infoset_t>(m, "Infoset")
      .def_property_readonly("actions", &infoset_t::actions)
      .def("__str__", &infoset_t::str);

  auto py_Game =
      py::class_<game_t>(m, "Game")
          .def("act", &game_t::act)
          .def("infoset", &game_t::infoset)
          .def_property_readonly("player", &game_t::player)
          .def("is_terminal", &game_t::is_terminal)
          .def("utility", &game_t::utility)
          .def("__copy__", &game_t::clone);

  py_Game.attr("Player") = py_Player;

  py::class_<flipguess_t>(m, "FlipGuess", py_Game)
      .def("heads", &flipguess_t::heads);

  py::class_<kuhn_poker_t>(m, "KuhnPoker", py_Game);

  py::class_<leduk_poker_t>(m, "LedukPoker", py_Game);

  py::class_<history_t>(m, "History")
      .def("act", &history_t::act)
      .def("infoset", &history_t::infoset)
      .def_property_readonly("player", &history_t::player)
      .def("is_terminal", &history_t::is_terminal)
      .def("utility", &history_t::utility)
      .def("__copy__", [](const history_t &h) { return history_t(h); });

  m.def("exploitability", &exploitability);

  py::class_<action_prob_t>(m, "ActionProb")
      .def_readwrite("a", &action_prob_t::a)
      .def_readwrite("pr_a", &action_prob_t::pr_a)
      .def_readwrite("rho1", &action_prob_t::rho1)
      .def_readwrite("rho2", &action_prob_t::rho2);

  py::class_<oos_t>(m, "OOS")
      .def(py::init<>())
      .def("search", &oos_t::search);

  py::class_<sigma_t>(m, "Sigma")
      .def("pr", &sigma_t::pr)
      .def("sample_pr", &sigma_t::sample_pr)
      .def("sample_eps", &sigma_t::sample_eps);

  m.def("make_py_sigma", [](py::object callback_fn) {
    return make_sigma<py_sigma_t>(move(callback_fn));
  });

  {
    using search_t = oos_t::search_t;
    using state_t = oos_t::search_t::state_t;

    auto py_OOS =
      py::class_<search_t>(m, "Search")
          .def(py::init<history_t, player_t>())
          .def_property_readonly("state", &search_t::state)
          .def("infoset", &search_t::infoset)
          .def("select", &search_t::select)
          .def("create", &search_t::create)
          .def("playout_step", &search_t::playout_step)
          .def("backprop", &search_t::backprop);

    py::enum_<oos_t::search_t::state_t>(py_OOS, "State")
        .value("SELECT", state_t::SELECT)
        .value("CREATE", state_t::CREATE)
        .value("PLAYOUT", state_t::PLAYOUT)
        .value("BACKPROP", state_t::BACKPROP)
        .value("FINISHED", state_t::FINISHED);
  }

  py::class_<tree_t>(m, "Tree")
      .def(py::init<>())
      .def("sigma_average", &tree_t::sigma_average)
      .def("size", &tree_t::size)
      .def("create_node", &tree_t::create_node)
      .def("lookup", (node_t &(tree_t::*)(const infoset_t &)) &tree_t::lookup)
      .def_property_readonly("nodes", (tree_t::map_t &(tree_t::*)()) &tree_t::nodes);

  py::class_<node_t>(m, "Node")
      .def("sigma_regret_matching", &node_t::sigma_regret_matching)
      .def("accumulate_regret", &node_t::accumulate_regret)
      .def("accumulate_average_strategy", &node_t::accumulate_average_strategy)
      .def_property_readonly("regret_n", (int (node_t::*)() const) &node_t::regret_n)
      .def_property_readonly("regrets", &node_t::regret_map)
      .def_property_readonly("average_strategy",
                             &node_t::average_strategy_map);

  py::class_<rng_t>(m, "Random")
      .def(py::init<>())
      .def(py::init<int>());

  auto py_Encoder =
      py::class_<encoder_t, batch_search_t::encoder_ptr_t>(m, "Encoder")
          .def("encoding_size", &encoder_t::encoding_size)
          .def("max_actions", &encoder_t::max_actions)
          .def("encode", &encoder_t::encode)
          .def("decode", &encoder_t::decode)
          .def("decode_and_sample", &encoder_t::decode_and_sample);

  py::class_<leduk_encoder_t>(m, "LedukEncoder", py_Encoder)
      .def(py::init<>());

  py::class_<batch_search_t>(m, "BatchSearch")
      .def(py::init<history_t, batch_search_t::encoder_ptr_t, int>())
      .def("generate_batch", &batch_search_t::generate_batch)
      .def("step", &batch_search_t::step)
      .def_property_readonly("tree", &batch_search_t::tree);

  m.def("make_flipguess", []() {
    return flipguess_t();
  });

  m.def("make_flipguess_history", []() {
    return make_history<flipguess_t>();
  });

  m.def("make_kuhn", []() {
    return kuhn_poker_t();
  });

  m.def("make_kuhn_history", []() {
    return make_history<kuhn_poker_t>();
  });

  m.def("make_leduk", []() {
    return leduk_poker_t();
  });

  m.def("make_leduk_history", []() {
    return make_history<leduk_poker_t>();
  });
}
