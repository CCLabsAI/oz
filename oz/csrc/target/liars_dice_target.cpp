#include "oos.h"

#include "liars_dice_target.h"

namespace oz {

using namespace std;

static auto cast_history(const history_t &h)
-> const liars_dice_t& {
  return h.cast<liars_dice_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const liars_dice_t::infoset_t& {
  return infoset.cast<liars_dice_t::infoset_t>();
}

auto liars_dice_target_t::target_actions(const infoset_t &target_infoset,
                                    const history_t &current_history) const
  -> set<action_t>
{
  using action_t = liars_dice_t::action_t;

  const auto &target_infoset_base = cast_infoset(target_infoset);
  const auto &current_game = cast_history(current_history);

 
 

  const auto &current_actions = current_game.history();
  const auto &target_actions = target_infoset_base.history;
  const auto next_ply_n = current_actions.size();

  // Q: should the history prefix match for targeting to work?

  if (current_actions.size() < target_actions.size()) {
    const auto target = target_actions[next_ply_n];

    Ensures(target != action_t::NextRound);

    Ensures(
        target == action_t::Raise_1face  ||
        target == action_t::Raise_2face  ||
        target == action_t::Raise_3face  ||
        target == action_t::Raise_4face  ||
        target == action_t::Raise_5face  ||
        target == action_t::Raise_0face  ||
        target == action_t::Raise_0dice  ||
        target == action_t::Raise_1dice  ||
        target == action_t::Raise_2dice  ||
        target == action_t::Raise_3dice  ||
        target == action_t::Raise_4dice  ||
        target == action_t::Call_liar);

    return { make_action(target) };
  }
  else {
    return { };
  }
}

} // namespace oz
