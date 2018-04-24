#include "oos.h"

#include "util.h"
#include "goofspiel2_target.h"

namespace oz {

using namespace std;
using card_t = goofspiel2_t::card_t;

static auto cast_history(const history_t &h) -> const goofspiel2_t& {
  return h.cast<goofspiel2_t>();
}

static auto other_player(player_t p) -> player_t {
  Expects(p == P1 || p == P2);
  return p == P1 ? P2 : P1;
}

template<typename T>
static auto set_without(const set<T> &s, T x) -> set<T> {
  auto t = set<T>(s);
  t.erase(x);
  return t;
}

static auto playable(const int turn,
                     const card_t card,
                     const set<card_t> &hand,
                     const player_t match_player,
                     const vector<card_t> &bids,
                     const vector<player_t> &wins) -> bool
{
  Expects(bids.size() == wins.size());

  // Base case: we have no more history remaining
  if ((unsigned) turn >= bids.size()) {
    return true;
  }

  bool card_playable = false;
  if (wins[turn] == CHANCE) {
    // If we know this turn was a draw, only one choice is possible
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
    const auto next_hand = set_without(hand, card);

    const auto playable_next = [&](const card_t &next_card) -> bool {
      return playable(turn + 1,
                      next_card, next_hand,
                      match_player, bids, wins);
    };

    return any_of(begin(next_hand), end(next_hand), playable_next);
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
  const auto &current_bids = current_game.bids(match_player);
  const auto &target_bids = target_game.bids(match_player);
  const auto &target_wins = target_game.wins();

  Expects(target_game.n_cards() == current_game.n_cards());

  const auto opponent = other_player(match_player);
  const auto &opponent_hand = current_game.hand(opponent);

  const auto next_turn = current_bids.size();

  if (current_bids.size() < target_bids.size()) {
    if (current_game.player() == match_player) {
      const auto target = target_bids[next_turn];
      return { make_action(target) };
    }
    else {
      const auto opponent_playable = [&](const card_t &card) -> bool {
        return playable(next_turn,
                        card, opponent_hand,
                        match_player, target_bids, target_wins);
      };

      auto actions = set<action_t> { };
      for (const auto &card : opponent_hand) {
        if (opponent_playable(card)) {
          actions.insert(make_action(card));
        }
      }

      Ensures(!actions.empty());
      return actions;
    }
  }

  return { };
}

game_t &goofspiel2_target_t::game() {
  return target_game;
}

} // namespace oz
