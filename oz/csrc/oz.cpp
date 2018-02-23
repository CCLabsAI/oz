#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kuhn.h"

namespace py = pybind11;


PYBIND11_MODULE(_ext, m) {
  m.doc() = "oz c++ extensions";

#ifdef VERSION_INFO
  m.attr("__version__") = py::str(VERSION_INFO);
#endif

  py::class_<KuhnPoker> py_kuhn_poker(m, "KuhnPoker");

  py_kuhn_poker
      .def(py::init<>())
      .def(py::init<const KuhnPoker&>())
      .def("is_terminal", &KuhnPoker::is_terminal)
      .def("act", &KuhnPoker::act)
      .def("legal_actions", &KuhnPoker::legal_actions)
      .def("reward", &KuhnPoker::reward)
      .def("infoset", (KuhnPoker::Infoset (KuhnPoker::*)(KuhnPoker::Player) const) &KuhnPoker::infoset)
      .def("infoset", (KuhnPoker::Infoset (KuhnPoker::*)() const) &KuhnPoker::infoset)
      .def_property_readonly("player", [](const KuhnPoker& self) {
          return self.player;
        })
      .def_property_readonly("showdown", [](const KuhnPoker& self) {
          return self.showdown;
        })
      .def_property_readonly("hand", [](const KuhnPoker& self) {
          return std::pair<KuhnPoker::Card,KuhnPoker::Card> {
            self.hand(KuhnPoker::Player::P1),
            self.hand(KuhnPoker::Player::P2)
          };
        })
      .def_property_readonly("pot", [](const KuhnPoker& self) {
          return std::pair<int,int> {
            self.pot(KuhnPoker::Player::P1),
            self.pot(KuhnPoker::Player::P2)
          };
        })
      .def_property_readonly("folded", [](const KuhnPoker& self) {
          return std::pair<bool,bool> {
            self.folded(KuhnPoker::Player::P1),
            self.folded(KuhnPoker::Player::P2)
          };
        })
      .def("__copy__", [](const KuhnPoker& self) {
          return KuhnPoker(self);
        });

  py::class_<KuhnPoker::Infoset>(py_kuhn_poker, "Infoset")
      .def("__str__", &KuhnPoker::Infoset::str)
      .def("__eq__",
        [](const KuhnPoker::Infoset& self, const KuhnPoker::Infoset& other) {
          return self == other;
        }
      )
      .def("__repr__",
        [](const KuhnPoker::Infoset &self) {
            return "<oz.KuhnPoker.Infoset '" + self.str() + "'>";
        }
      );


  py::enum_<KuhnPoker::Action>(py_kuhn_poker, "Action")
      .value("Pass", KuhnPoker::Action::Pass)
      .value("Bet", KuhnPoker::Action::Bet)
      .value("JQ", KuhnPoker::Action::JQ)
      .value("JK", KuhnPoker::Action::JK)
      .value("QJ", KuhnPoker::Action::QJ)
      .value("QK", KuhnPoker::Action::QK)
      .value("KJ", KuhnPoker::Action::KJ)
      .value("KQ", KuhnPoker::Action::KQ);

  py::enum_<KuhnPoker::Card>(py_kuhn_poker, "Card")
      .value("NA", KuhnPoker::Card::NA)
      .value("Jack", KuhnPoker::Card::Jack)
      .value("Queen", KuhnPoker::Card::Queen)
      .value("King", KuhnPoker::Card::King);

  py::enum_<KuhnPoker::Player>(py_kuhn_poker, "Player")
      .value("Chance", KuhnPoker::Player::Chance)
      .value("P1", KuhnPoker::Player::P1)
      .value("P2", KuhnPoker::Player::P2);
}
