#include "oos.h"

#include "util.h"
#include "goofspiel2_target.h"

namespace oz {

using namespace std;

auto goofspiel2_target_t::cast_history(const history_t &h)
  -> const goofspiel2_t&
{
  return h.cast<goofspiel2_t>();
}

static auto other_player(player_t p) -> player_t {
  switch(p) {
    case P1:
      return P2;
    case P2:
      return P1;
    case CHANCE:
      return CHANCE;
    default:
      return p;
  }
}

auto goofspiel2_target_t::target_actions(const history_t &current_history) const
  -> set<action_t>
{
  Expects(!target_game.is_terminal());

  const auto match_player = this->match_player;
  const auto &current_game = cast_history(current_history);

  const auto &current_bids = current_game.bids(match_player);
  const auto &target_bids = target_game.bids(match_player);
  const auto next_turn_n = current_bids.size();

  if (current_game.player() == match_player) {
    if (current_bids.size() < target_bids.size()) {
      const auto target = target_bids[next_turn_n];
      return { make_action(target) };
    }
    else {
      return { };
    }
  }
  else {
    // NB There are no public actions in II Goofspiel and we
    // only know the outcome of the bid, not the card that
    // was bid, we neet to make sure the opponent bids a card
    // consistent with the observed outcomes, which requires
    // seem to require reasonably sophisticated constraint solving
    if (current_bids.size() < target_bids.size()) {
      const auto &wins = target_game.wins();
      const auto winner = wins[next_turn_n];

      if (winner == CHANCE) { // Draw means opponent played the same number
        const auto target = target_bids[next_turn_n];
        return { make_action(target) };
      }

      auto cards = set<goofspiel2_t::card_t> { };
      auto actions = set<action_t> { };
      auto card_ins = inserter(cards, end(cards));
      auto action_ins = inserter(actions, end(actions));
      auto target_bid = target_bids[next_turn_n];
      auto &other_hand = current_game.hand(other_player(match_player));

      if (winner == match_player) { // opponent player smaller number
        copy_if(begin(other_hand), end(other_hand), card_ins,
                [&](const auto& card) { return card < target_bid; });
      }

      if (winner == other_player(match_player)) {
        copy_if(begin(other_hand), end(other_hand), card_ins,
                [&](const auto& card) { return card > target_bid; });
      }

      transform(begin(cards), end(cards), action_ins,
                [](const auto& card) { return make_action(card); });

      Expects(!actions.empty());

      return actions;
    }
  }

  return { };
}

game_t &goofspiel2_target_t::game() {
  return target_game;
}

} // namespace oz
