#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rock_paper_scissors.h"
#include "flip_guess.h"
#include "kuhn.h"

namespace py = pybind11;


template<class Game> py::module def_game(py::module m, const char* name);

void def_rock_paper_scissors(py::module m);
void def_kuhn_poker(py::module m);
void def_flip_guess(py::module m);

PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#endif

  def_rock_paper_scissors(m);
  def_kuhn_poker(m);
  def_flip_guess(m);
}

template<class Game>
py::module def_game(py::module m, const char* name) {
  using Player = typename Game::Player;
  using Infoset = typename Game::Infoset;

  py::class_<Game> m_game(m, name);

  m_game
      .def(py::init<>())
      .def(py::init<const Game&>())
      .def("__copy__", [](const Game& self) { return Game(self); })
      .def("is_terminal", &Game::is_terminal)
      .def("act", &Game::act)
      .def("legal_actions", &Game::legal_actions)
      .def("reward", &Game::reward)
      .def("infoset", (Infoset (Game::*)(Player) const) &Game::infoset)
      .def("infoset", (Infoset (Game::*)() const) &Game::infoset)
      .def_property_readonly("player", [](const Game& self) {
          return self.player;
        });

  py::class_<Infoset>(m_game, "Infoset")
      .def("__str__", &Infoset::str)
      .def("__eq__", [](const Infoset& self, const Infoset& other) {
          return self == other;
        })
      .def("__hash__", [](const Infoset& self) {
          return py::hash(py::str(self.str()));
        });

  py::enum_<Player>(m_game, "Player")
      .value("Chance", Player::Chance)
      .value("P1", Player::P1)
      .value("P2", Player::P2);

  return m_game;
}

void def_flip_guess(py::module m) {
  using Action = FlipGuess::Action;

  py::module m_game = def_game<FlipGuess>(m, "FlipGuess");

  py::enum_<Action>(m_game, "Action")
      .value("Heads", Action::Heads)
      .value("Tails", Action::Tails)
      .value("Left", Action::Left)
      .value("Right", Action::Right);
}

void def_rock_paper_scissors(py::module m) {
  using Action = RockPaperScissors::Action;

  py::module m_rps = def_game<RockPaperScissors>(m, "RockPaperScissors");

  py::enum_<Action>(m_rps, "Action")
      .value("Rock", Action::Rock)
      .value("Paper", Action::Paper)
      .value("Scissors", Action::Scissors);
}

void def_kuhn_poker(py::module m) {
  using Player = KuhnPoker::Player;
  using Action = KuhnPoker::Action;
  using Infoset = KuhnPoker::Infoset;
  using Card = KuhnPoker::Card;

  py::class_<KuhnPoker> py_kuhn_poker(m, "KuhnPoker");

  py_kuhn_poker
      .def(py::init<>())
      .def(py::init<const KuhnPoker&>())
      .def("__copy__", [](const KuhnPoker& self) { return KuhnPoker(self); })
      .def("is_terminal", &KuhnPoker::is_terminal)
      .def("act", &KuhnPoker::act)
      .def("legal_actions", &KuhnPoker::legal_actions)
      .def("reward", &KuhnPoker::reward)
      .def("infoset", (Infoset (KuhnPoker::*)(Player) const) &KuhnPoker::infoset)
      .def("infoset", (Infoset (KuhnPoker::*)() const) &KuhnPoker::infoset)
      .def_property_readonly("player", [](const KuhnPoker& self) {
          return self.player;
        })
      .def_property_readonly("showdown", [](const KuhnPoker& self) {
          return self.showdown;
        })
      .def_property_readonly("hand", [](const KuhnPoker& self) {
          return py::make_tuple(
            self.hand(Player::P1),
            self.hand(Player::P2)
          );
        })
      .def_property_readonly("pot", [](const KuhnPoker& self) {
          return py::make_tuple(
            self.pot(Player::P1),
            self.pot(Player::P2)
          );
        })
      .def_property_readonly("folded", [](const KuhnPoker& self) {
          return py::make_tuple(
            self.folded(Player::P1),
            self.folded(Player::P2)
          );
        });

  py::class_<Infoset>(py_kuhn_poker, "Infoset")
      .def("__str__", &Infoset::str)
      .def("__eq__", [](const Infoset& self, const Infoset& other) {
          return self == other;
        })
      .def("__hash__", [](const Infoset& self) {
          return py::hash(py::str(self.str()));
        });

  py::enum_<Player>(py_kuhn_poker, "Player")
      .value("Chance", Player::Chance)
      .value("P1", Player::P1)
      .value("P2", Player::P2);

  py::enum_<Action>(py_kuhn_poker, "Action")
      .value("Pass", Action::Pass)
      .value("Bet", Action::Bet)
      .value("JQ", Action::JQ)
      .value("JK", Action::JK)
      .value("QJ", Action::QJ)
      .value("QK", Action::QK)
      .value("KJ", Action::KJ)
      .value("KQ", Action::KQ);

  py::enum_<Card>(py_kuhn_poker, "Card")
      .value("NA", Card::NA)
      .value("Jack", Card::Jack)
      .value("Queen", Card::Queen)
      .value("King", Card::King);
}
