#include "oos.h"

#include "leduk_target.h"

namespace oz {

using namespace std;

static auto cast_history(const history_t &h)
-> const leduk_poker_t& {
  return h.cast<leduk_poker_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const leduk_poker_t::infoset_t& {
  return infoset.cast<leduk_poker_t::infoset_t>();
}

auto leduk_target_t::target_actions(const infoset_t &target_infoset,
                                    const history_t &current_history) const
  -> set<action_t>
{
  using card_t = leduk_poker_t::card_t;
  using action_t = leduk_poker_t::action_t;

  const auto &target_infoset_base = cast_infoset(target_infoset);
  const auto &current_game = cast_history(current_history);

  // NB returning an empty set here means no targeting
  // TODO create custom return type

  // TODO clean this up
  if (current_game.player() == CHANCE) {
    if (current_game.hand(P1) != card_t::NA &&
        current_game.hand(P2) != card_t::NA &&
        current_game.board() == card_t::NA) {
      switch (target_infoset_base.board) {
        case card_t::Jack:
          return { make_action(action_t::J) };
        case card_t::Queen:
          return { make_action(action_t::Q) };
        case card_t::King:
          return { make_action(action_t::K) };
        default:
          return { };
      }
    }
  }

  const auto &current_actions = current_game.history();
  const auto &target_actions = target_infoset_base.history;
  const auto next_ply_n = current_actions.size();

  // Q: should the history prefix match for targeting to work?

  if (current_actions.size() < target_actions.size()) {
    const auto target = target_actions[next_ply_n];

    Ensures(target != action_t::NextRound);

    Ensures(
        target == action_t::Raise ||
        target == action_t::Call  ||
        target == action_t::Fold);

    return { make_action(target) };
  }
  else {
    return { };
  }
}

} // namespace oz
