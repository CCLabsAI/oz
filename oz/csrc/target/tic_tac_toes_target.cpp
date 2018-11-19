#include "oos.h"

#include "tic_tac_toes_target.h"

namespace oz {

using namespace std;

static auto cast_history(const history_t &h)
-> const tic_tac_toes_t& {
  return h.cast<tic_tac_toes_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const tic_tac_toes_t::infoset_t& {
  return infoset.cast<tic_tac_toes_t::infoset_t>();
}

  auto tic_tac_toes_target_t::target_actions(const infoset_t &target_infoset,const history_t &current_history) const
  -> set<action_t>
{
  using action_t = tic_tac_toes_t::action_t;

  const auto &target_infoset_base = cast_infoset(target_infoset);
  const auto &current_game = cast_history(current_history);
  
  const auto &current_actions = current_game.history();
  const auto &target_actions = target_infoset_base.history;
  const auto next_ply_n = current_actions.size();

  if (current_actions.size() < target_actions.size()) {
    const auto target = target_actions[next_ply_n];

    Ensures(target != action_t::NextRound);

    Ensures(
        target == action_t::fill_1 ||
        target == action_t::fill_2  ||
        target == action_t::fill_3  ||
        target == action_t::fill_4  ||
        target == action_t::fill_5  ||
        target == action_t::fill_6  ||
        target == action_t::fill_7  ||
        target == action_t::fill_8  ||
        target == action_t::fill_9);

    return { make_action(target) };
  }
  else {
    return { };
  }
}

} // namespace oz
