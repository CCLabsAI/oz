#include "holdem.h"

namespace oz {

using namespace std;

void holdem_poker_t::act_(action_t a) {
  throw std::logic_error("not implemented");
}

auto holdem_poker_t::infoset() const -> oz::infoset_t {
  throw std::logic_error("not implemented");
};

bool holdem_poker_t::is_terminal() const {
  throw std::logic_error("not implemented");
}

value_t holdem_poker_t::utility(player_t player) const {
  throw std::logic_error("not implemented");
}

auto holdem_poker_t::chance_actions() const -> action_prob_map_t {
  throw std::logic_error("not implemented");
}

auto holdem_poker_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t {
  throw std::logic_error("not implemented");
}

auto holdem_poker_t::chance_actions(action_prob_allocator_t alloc) const
  -> action_prob_map_t {
  throw std::logic_error("not implemented");
}

auto holdem_poker_t::str() const -> std::string {
  throw std::logic_error("not implemented");
};

} // namespace oz
