#include "target.h"

namespace oz {

using namespace std;

auto leduk_target_t::cast_history(const history_t &h) -> const leduk_poker_t& {
  return h.cast<leduk_poker_t>();
}

auto leduk_target_t::target_actions(const history_t &current_history,
                                    const history_t &target_history,
                                    vector<action_t> actions) const
  -> set<action_t>
{
  const auto &current_game = cast_history(current_history);
  const auto &target_game = cast_history(target_history);

  // TODO clean this up
  if (current_game.player() == CHANCE) {
    if (current_game.hand(P1) != leduk_poker_t::card_t::NA &&
        current_game.hand(P2) != leduk_poker_t::card_t::NA &&
        current_game.board() == leduk_poker_t::card_t::NA) {
      switch (target_game.board()) {
        case leduk_poker_t::card_t::Jack:
          return { make_action(leduk_poker_t::action_t::J) };
        case leduk_poker_t::card_t::Queen:
          return { make_action(leduk_poker_t::action_t::Q) };
        case leduk_poker_t::card_t::King:
          return { make_action(leduk_poker_t::action_t::K) };
        default:
          return set<action_t>(begin(actions), end(actions));
      }
    }
  }

  const auto &current_actions = current_game.history();
  const auto &target_actions = target_game.history();
  const auto next_ply_n = current_actions.size();

  // Q: should the history prefix match for targeting to work?

  if (current_actions.size() < target_actions.size()) {
    const auto target = target_actions[next_ply_n];
    Ensures(target != leduk_poker_t::action_t::NextRound);

    Ensures(
        target == leduk_poker_t::action_t::Raise ||
        target == leduk_poker_t::action_t::Call  ||
        target == leduk_poker_t::action_t::Fold);

    return { make_action(target) };
  }
  else {
    return set<action_t>(begin(actions), end(actions));
  }
}

} // namespace oz
