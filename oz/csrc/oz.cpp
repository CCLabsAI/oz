#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "game.h"
#include "best_response.h"
#include "oos.h"
#include "batch.h"
#include "encoder.h"
#include "mcts.h"
#include "mcts_batch.h"
#include "py_sigma.h"
#include "py_sigma_batch.h"
#include "games/flipguess.h"
#include "games/kuhn.h"
#include "games/leduk.h"
#include "encoder/leduk_encoder.h"
#include "target/leduk_target.h"
#include "games/goofspiel2.h"
#include "target/goofspiel2_target.h"
#include "encoder/goofspiel2_encoder.h"

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

// Look at this handsome template instantiation right here...
template <typename Key, typename Value, typename Compare, typename Alloc>
struct pybind11::detail::type_caster<boost::container::flat_map<Key, Value, Compare, Alloc>>
  : map_caster<boost::container::flat_map<Key, Value, Compare, Alloc>, Key, Value> { };

void bind_oz(py::module &m) {
  using namespace oz;
  using std::begin;
  using std::end;

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

  m.def("make_action_raw", [](int n) {
    return make_action(n);
  });

  py::class_<infoset_t>(m, "Infoset")
      .def_property_readonly("actions", [](const infoset_t &self) -> vector<action_t> {
        const auto &actions = self.actions();
        return vector<action_t>(begin(actions), end(actions));
      })
      .def("__str__", &infoset_t::str);

  auto py_Game =
      py::class_<game_t>(m, "Game")
          .def("act", &game_t::act)
          .def("infoset", (infoset_t (game_t::*)() const) &game_t::infoset)
          .def_property_readonly("player", &game_t::player)
          .def("is_terminal", &game_t::is_terminal)
          .def("utility", &game_t::utility)
          .def("__str__", &game_t::str)
          .def("__copy__", &game_t::clone);

  py_Game.attr("Player") = py_Player;

  py::class_<flipguess_t>(m, "FlipGuess", py_Game)
      .def("heads", &flipguess_t::heads);

  py::class_<kuhn_poker_t>(m, "KuhnPoker", py_Game);

  auto py_LedukPoker =
    py::class_<leduk_poker_t>(m, "LedukPoker", py_Game)
        .def("hand", (leduk_poker_t::card_t (leduk_poker_t::*)(player_t) const) &leduk_poker_t::hand)
        .def("board", &leduk_poker_t::board)
        .def("pot", (int (leduk_poker_t::*)(player_t) const) &leduk_poker_t::pot)
        .def("folded", (bool (leduk_poker_t::*)(player_t) const) &leduk_poker_t::folded);

  py::enum_<leduk_poker_t::card_t>(py_LedukPoker, "Card")
      .value("NA", leduk_poker_t::card_t::NA)
      .value("Jack", leduk_poker_t::card_t::Jack)
      .value("Queen", leduk_poker_t::card_t::Queen)
      .value("King", leduk_poker_t::card_t::King);

  py::class_<goofspiel2_t>(m, "Goofspiel2", py_Game)
    .def("score", (int (goofspiel2_t::*)(player_t p) const) &goofspiel2_t::score)
    .def_property_readonly("turn", &goofspiel2_t::turn)
    .def("hand", [](const goofspiel2_t &self, player_t p) -> set<goofspiel2_t::card_t> {
      const auto &hand = self.hand(p);
      set<goofspiel2_t::card_t> s;
      for(goofspiel2_t::card_t card = 0; card < hand.size(); card++) {
        s.insert(card);
      }
      return s;
    })
    // .def("bids", (const goofspiel2_t::bids_t &(goofspiel2_t::*)(player_t p) const) &goofspiel2_t::bids)
    .def("bids", [](const goofspiel2_t &self, player_t p) -> vector<goofspiel2_t::card_t> {
      auto v = self.bids(p);
      return vector<goofspiel2_t::card_t>(begin(v), end(v));
    })
    .def_property_readonly("wins", [](const goofspiel2_t &self) -> vector<player_t> {
      auto v = self.wins();
      return vector<player_t>(begin(v), end(v));
    });

  py::class_<history_t>(m, "History")
      .def("act", &history_t::act)
      .def("infoset", (infoset_t (history_t::*)() const) &history_t::infoset)
      .def_property_readonly("player", &history_t::player)
      .def("is_terminal", &history_t::is_terminal)
      .def("utility", &history_t::utility)
      .def("sample_chance", &history_t::sample_chance)
      .def_property_readonly("game", // TODO figure out if there is a better way
        [](const history_t &self) -> const game_t& {
          return self.cast<game_t>();
        }, py::return_value_policy::reference_internal)
      .def("__str__", &history_t::str)
      .def("__copy__", [](const history_t &self) { return history_t(self); });

  m.def("exploitability", &exploitability);
  m.def("gebr", &gebr);

  py::class_<action_prob_t>(m, "ActionProb")
      .def_readwrite("a", &action_prob_t::a)
      .def_readwrite("pr_a", &action_prob_t::pr_a)
      .def_readwrite("rho1", &action_prob_t::rho1)
      .def_readwrite("rho2", &action_prob_t::rho2);

  // NB this needs to be before OOS, because of the default argument
  // TODO clean up the api so this isn't necessary
  py::class_<target_t>(m, "Target");

  py::class_<oos_t>(m, "OOS")
      .def(py::init<>())
      .def("reset_targeting_ratio", &oos_t::reset_targeting_ratio)
      .def_property_readonly("avg_targeting_ratio", &oos_t::avg_targeting_ratio)
      .def("search", &oos_t::search,
           py::arg("history"),
           py::arg("n_iter"),
           py::arg("tree"),
           py::arg("rng"),
           py::arg("eps") = 0.4,
           py::arg("delta") = 0.9,
           py::arg("gamma") = 0.00,
           py::arg("beta") = 1.0)
      .def("search_targeted", &oos_t::search_targeted,
           py::arg("history"),
           py::arg("n_iter"),
           py::arg("tree"),
           py::arg("rng"),
           py::arg("target"),
           py::arg("target_infoset"),
           py::arg("eps") = 0.4,
           py::arg("delta") = 0.9,
           py::arg("gamma") = 0.01,
           py::arg("beta") = 0.99);

  auto py_Sigma =
    py::class_<sigma_t>(m, "Sigma")
        .def("pr", &sigma_t::pr)
        .def("sample_pr", &sigma_t::sample_pr)
        .def("sample_eps", &sigma_t::sample_eps);

  m.def("make_py_sigma", [](py::object callback_fn) {
    return make_sigma<py_sigma_t>(move(callback_fn));
  });

  py::class_<py_sigma_batch_t>(m, "SigmaBatch")
    .def(py::init<>())
    .def("walk_infosets", &py_sigma_batch_t::walk_infosets)
    .def("generate_batch", &py_sigma_batch_t::generate_batch)
    .def("store_probs", &py_sigma_batch_t::store_probs)
    .def("make_sigma", &py_sigma_batch_t::make_sigma);

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
      .def("sigma_average", &tree_t::sigma_average,
        py::return_value_policy::move,
        py::keep_alive<0, 1>())
      .def("size", &tree_t::size)
      .def("clear", &tree_t::clear)
      .def("create_node", &tree_t::create_node)
      .def("lookup", (node_t &(tree_t::*)(const infoset_t &)) &tree_t::lookup)
      .def_property_readonly("nodes", (tree_t::map_t &(tree_t::*)()) &tree_t::nodes);

  py::class_<node_t>(m, "Node")
      .def("accumulate_regret", &node_t::accumulate_regret)
      .def("accumulate_average_strategy", &node_t::accumulate_average_strategy)
      .def_property_readonly("regret_n", (int (node_t::*)() const) &node_t::regret_n)
      .def_property_readonly("regrets", &node_t::regret_map)
      .def_property_readonly("average_strategy",
                             &node_t::average_strategy_map);

  py::class_<rng_t>(m, "Random")
      .def(py::init<int>())
      .def(py::init<>([]() -> rng_t {
        std::random_device r;
        std::seed_seq seed { r(), r(), r(), r(), r(), r(), r(), r() };
        return rng_t(seed);
      }));

  auto py_Encoder =
      py::class_<encoder_t, std::shared_ptr<encoder_t>>(m, "Encoder")
          .def("encoding_size", &encoder_t::encoding_size)
          .def("max_actions", &encoder_t::max_actions)
          .def("encode", &encoder_t::encode)
          .def("encode_sigma", &encoder_t::encode_sigma)
          .def("decode", &encoder_t::decode)
          .def("decode_and_sample", &encoder_t::decode_and_sample);

  py::class_<leduk_encoder_t,
             std::shared_ptr<leduk_encoder_t>>(m, "LedukEncoder", py_Encoder);

  py::class_<goofspiel2_encoder_t,
             std::shared_ptr<goofspiel2_encoder_t>>(m, "Goofspiel2Encoder", py_Encoder);

  py::class_<batch_search_t>(m, "BatchSearch")
      .def(py::init<int, history_t, std::shared_ptr<encoder_t>>())
      .def(py::init<int, history_t, std::shared_ptr<encoder_t>, target_t,
                    prob_t, prob_t, prob_t, prob_t, prob_t>(),
           py::arg("batch_size"),
           py::arg("history"),
           py::arg("encoder"),
           py::arg("target"),
           py::arg("eps"),
           py::arg("delta"),
           py::arg("gamma"),
           py::arg("beta"),
           py::arg("eta"))
      .def("generate_batch", &batch_search_t::generate_batch)
      .def("step", (void (batch_search_t::*)(at::Tensor probs, rng_t &rng)) &batch_search_t::step)
      .def("step", (void (batch_search_t::*)(rng_t &rng)) &batch_search_t::step)
      .def("target", &batch_search_t::target)
      .def("reset_targeting_ratio", &batch_search_t::reset_targeting_ratio)
      .def_property_readonly("tree", &batch_search_t::tree,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("avg_targeting_ratio",
                             &batch_search_t::avg_targeting_ratio);

  py::class_<mcts::params_t>(m, "MCTSParams")
      .def(py::init<>())
      .def_readwrite("c", &mcts::params_t::c)
      .def_readwrite("gamma", &mcts::params_t::gamma)
      .def_readwrite("eta", &mcts::params_t::eta)
      .def_readwrite("d", &mcts::params_t::d)
      .def_readwrite("smooth", &mcts::params_t::smooth)
      .def_readwrite("search_player", &mcts::params_t::search_player);

  py::class_<mcts::tree_t>(m, "MCTSTree")
      .def(py::init<>())
      .def("sigma_average", &mcts::tree_t::sigma_average,
           py::return_value_policy::move,
           py::keep_alive<0, 1>())
//      .def("lookup", (mcts::node_t &(mcts::tree_t::*)(const infoset_t &)) &mcts::tree_t::lookup)
      .def_readwrite("nodes", &mcts::tree_t::nodes);

  py::class_<mcts::batch_search_t>(m, "MCTSBatchSearch")
      .def(py::init<int, history_t, std::shared_ptr<encoder_t>, mcts::params_t>())
      .def("generate_batch", &mcts::batch_search_t::generate_batch)
      .def("step", (void (mcts::batch_search_t::*)(at::Tensor probs, rng_t &rng)) &mcts::batch_search_t::step)
      .def("step", (void (mcts::batch_search_t::*)(rng_t &rng)) &mcts::batch_search_t::step)
      .def_property_readonly("tree", &mcts::batch_search_t::tree,
                             py::return_value_policy::reference_internal);


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

  m.def("make_leduk_encoder", []() {
    return std::make_shared<leduk_encoder_t>();
  });

  m.def("make_leduk_target", []() {
    return make_target<leduk_target_t>();
  });

  m.def("make_goofspiel2", [](int n) {
    return goofspiel2_t(n);
  });

  m.def("make_goofspiel2_history", [](int n) {
    return make_history<goofspiel2_t>(n);
  });

  m.def("make_goofspiel2_target", []() {
    return make_target<goofspiel2_target_t>();
  });

  m.def("make_goofspiel2_encoder", [](int n) {
    return std::make_shared<goofspiel2_encoder_t>(n);
  });

}
