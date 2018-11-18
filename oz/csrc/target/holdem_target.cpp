#include "oos.h"

#include "holdem_target.h"

namespace oz {

using namespace std;

static auto cast_history(const history_t &h)
-> const holdem_poker_t& {
  return h.cast<holdem_poker_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const holdem_poker_t::infoset_t& {
  return infoset.cast<holdem_poker_t::infoset_t>();
}

auto holdem_target_t::target_actions(const infoset_t &target_infoset,
                                     const history_t &current_history) const
  -> set<action_t>
{
  using action_t = holdem_poker_t::action_t;
  using phase_t = holdem_poker_t::phase_t;

  const auto &target_infoset_base = cast_infoset(target_infoset);
  const auto &current_game = cast_history(current_history);

  // NB returning an empty set here means no targeting
  if (current_game.phase() == phase_t::DEAL_BOARD) {
    auto next_board_index = current_game.board().size();
    auto card = target_infoset_base.board[next_board_index];
    auto a = make_action(holdem_poker_t::deal_action_for_card(card));
    return { a };
  }

  const auto &current_actions = current_game.history();
  const auto &target_actions = target_infoset_base.history;
  const auto next_ply_n = current_actions.size();

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
