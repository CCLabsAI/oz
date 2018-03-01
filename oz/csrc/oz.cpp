#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rock_paper_scissors.h"
#include "kuhn.h"

namespace py = pybind11;

void def_rock_paper_scissors(py::module m);
void def_kuhn_poker(py::module m);

PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#endif

  def_rock_paper_scissors(m);
  def_kuhn_poker(m);
}

void def_rock_paper_scissors(py::module m) {
  using Player = RockPaperScissors::Player;
  using Action = RockPaperScissors::Action;
  using Infoset = RockPaperScissors::Infoset;

  py::class_<RockPaperScissors> py_rps(m, "RockPaperScissors");

  py_rps
      .def(py::init<>())
      .def(py::init<const RockPaperScissors&>())
      .def("__copy__", [](const RockPaperScissors& self) { return RockPaperScissors(self); })
      .def("is_terminal", &RockPaperScissors::is_terminal)
      .def("act", &RockPaperScissors::act)
      .def("legal_actions", &RockPaperScissors::legal_actions)
      .def("reward", &RockPaperScissors::reward)
      .def("infoset", (Infoset (RockPaperScissors::*)(Player) const) &RockPaperScissors::infoset)
      .def("infoset", (Infoset (RockPaperScissors::*)() const) &RockPaperScissors::infoset)
      .def_property_readonly("player", [](const RockPaperScissors& self) {
          return self.player;
        });

  py::class_<Infoset>(py_rps, "Infoset")
      .def("__str__", &Infoset::str)
      .def("__eq__", [](const Infoset& self, const Infoset& other) {
          return self == other;
        })
      .def("__hash__", [](const Infoset& self) {
          return py::hash(py::str(self.str()));
        });

  py::enum_<Action>(py_rps, "Action")
      .value("Rock", Action::Rock)
      .value("Paper", Action::Paper)
      .value("Scissors", Action::Scissors);

  py::enum_<Player>(py_rps, "Player")
      .value("Chance", Player::Chance)
      .value("P1", Player::P1)
      .value("P2", Player::P2);
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
