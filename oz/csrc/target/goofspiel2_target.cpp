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

static auto playable(int turn,
                     goofspiel2_t::card_t card,
                     set<goofspiel2_t::card_t> hand, int n_cards,
                     player_t match_player,
                     const vector<goofspiel2_t::card_t> &bids,
                     const vector<player_t> &wins) -> bool
{
  Expects(bids.size() == wins.size());

  // Base case: we have no more history to be consistent with.
  const auto max_turn = static_cast<int>(bids.size());
  if (turn >= max_turn) {
    return true;
  }

  bool card_playable;
  if (wins[turn] == CHANCE) {
    // If we know this turn was a draw, only once choice is possible
    card_playable = (card == bids[turn]);
  }
  else if (wins[turn] == match_player) {
    // If we know this was a win, card value must be lower than bid
    card_playable = (card < bids[turn]);
  }
  else {
    // If we know this was a loss, card value must be higher than bid
    card_playable = (card > bids[turn]);
  }

  if (card_playable) {
    // Recursive case: at least one card must be playable next turn
    bool next_playable = false;

    for (const auto &next_card : hand) {
      set<goofspiel2_t::card_t> next_hand(hand);
      next_hand.erase(next_card);

      next_playable = playable(turn+1,
                               next_card, next_hand, n_cards,
                               match_player, bids, wins);

      if (next_playable) break;
    }

    return next_playable;
  }
  else {
    return false;
  }
}

auto goofspiel2_target_t::target_actions(const history_t &current_history) const
  -> set<action_t>
{
  Expects(!target_game.is_terminal());

  const auto &current_game = cast_history(current_history);

  Expects(target_game.n_cards() == current_game.n_cards());

  const auto &current_bids = current_game.bids(match_player);
  const auto &target_bids = target_game.bids(match_player);
  const auto &target_wins = target_game.wins();
  const auto next_turn_n = static_cast<int>(current_bids.size());
  const auto opponent = other_player(match_player);
  const auto &opponent_hand = current_game.hand(opponent);

  if (current_bids.size() < target_bids.size()) {
    if (current_game.player() == match_player) {
      const auto target = target_bids[next_turn_n];
      return { make_action(target) };
    }
    else {
      auto s = set<action_t> { };

      for (const auto &card : opponent_hand) {
        set<goofspiel2_t::card_t> remaining_hand(opponent_hand);
        remaining_hand.erase(card);

        bool b = playable(next_turn_n,
                          card, remaining_hand, target_game.n_cards(),
                          match_player, target_bids, target_wins);

        if (b) {
          s.insert(make_action(card));
        }
      }

      Expects(!s.empty());

      return s;
    }
  }

  return { };
}

game_t &goofspiel2_target_t::game() {
  return target_game;
}

} // namespace oz
