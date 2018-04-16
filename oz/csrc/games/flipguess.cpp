#include <typeinfo>
#include <cassert>

#include "flipguess.h"

namespace oz {

using namespace std;

auto flipguess_t::infoset() const -> oz::infoset_t {
  Expects(player() != CHANCE);
  return make_infoset<infoset_t>(player_);
}

auto flipguess_t::utility(player_t player) const -> value_t {
  assert (is_terminal());
  value_t u = 0;

  if (heads_ && p1_action_ == action_t::Left) {
    u = 1;
  }
  else {
    if (p1_action_ == p2_action_) {
      u = 3;
    }
    else {
      u = 0;
    }
  }

  return relative_utility(player, u);
}

void flipguess_t::act_(flipguess_t::action_t a) {
  if (player_ == CHANCE) {
    if (a == action_t::Heads) {
      heads_ = true;
    }

    player_ = P1;
  }
  else if (player_ == P1) {
    if (heads_) {
      finished_ = true;
    }

    p1_action_ = a;
    player_ = P2;
  }
  else if (player_ == P2) {
    p2_action_ = a;
    finished_ = true;
  }
}

auto flipguess_t::infoset_t::actions() const -> vector<oz::action_t> {
  static const vector<oz::action_t> player_actions {
      make_action(action_t::Left),
      make_action(action_t::Right),
  };

  return player_actions;
}

auto flipguess_t::infoset_t::str() const -> string {
  if (player_ == CHANCE) {
    return "Chance";
  }
  else if (player_ == P1) {
    return "P1";
  }
  else if (player_ == P2) {
    return "P2";
  }
  else {
    return "?";
  }
}

bool flipguess_t::infoset_t::is_equal(const oz::infoset_t::concept_t& that)
const {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const flipguess_t::infoset_t&>(that);
    return player_ == that_.player_;
  }
  else {
    return false;
  }
}

size_t flipguess_t::infoset_t::hash() const {
  return std::hash<player_t>()(player_);
}

auto flipguess_t::chance_actions() const -> map<oz::action_t, prob_t> {
  Expects(player() == CHANCE);

  static const map<oz::action_t, prob_t> actions {
    { make_action(action_t::Heads), 0.5 },
    { make_action(action_t::Tails), 0.5 }
  };

  return actions;
}

} // namespace oz
