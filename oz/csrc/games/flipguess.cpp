#include <typeinfo>

#include "flipguess.h"

namespace oz {

using namespace std;

auto flipguess_t::act(oz::action_t a) -> void {
  auto a_native = static_cast<action_t>(a.index());
  act_(a_native);
}

auto flipguess_t::infoset() const -> oz::infoset_t {
  return make_infoset<infoset_t>(player_);
}

value_t relative_utility(player_t player, value_t u) {
  return player == P2 ? -u : u;
}

auto flipguess_t::utility(player_t player) const -> value_t {
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

auto flipguess_t::act_(flipguess_t::action_t a) -> void {
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
  if (player_ == CHANCE) {
    return vector<oz::action_t> {
        oz::action_t(static_cast<int>(action_t::Heads)),
        oz::action_t(static_cast<int>(action_t::Tails)),
    };
  }
  else {
    return vector<oz::action_t> {
        oz::action_t(static_cast<int>(action_t::Left)),
        oz::action_t(static_cast<int>(action_t::Right)),
    };
  }
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

auto flipguess_t::infoset_t::is_equal(const oz::infoset_t::concept_t& that)
const -> bool {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const flipguess_t::infoset_t&>(that);
    return player_ == that_.player_;
  }
  else {
    return false;
  }
}

} // namespace oz
