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

static bool consistent(player_t player,
                      vector<bool> &opponent_cards,
                      int turn,
                      int max_turn,
                      const vector <goofspiel2_t::card_t> &target_bids,
                      const vector <player_t> &wins)
{
  if (turn >= max_turn) {
    return true;
  }

  auto known_bid = target_bids[turn];

  // This was a draw, opponent must have played specific card
  if (wins[turn] == CHANCE) {
    if(!opponent_cards[known_bid]) {
      return false;
    }
    else {
      opponent_cards[known_bid] = false;
      auto b = consistent(player,
                          opponent_cards,
                          turn + 1,
                          max_turn,
                          target_bids,
                          wins);
      opponent_cards[known_bid] = true;
      return b;
    }
  }

  // player won, lower card must be playable
  if (wins[turn] == player) {
    for (int i = known_bid - 1; i > 0; --i) {
      if (!opponent_cards[known_bid]) {
        continue;
      }
      else {
        opponent_cards[known_bid] = false;
        auto b =
            consistent(player,
                       opponent_cards,
                       turn + 1,
                       max_turn,
                       target_bids,
                       wins);
        opponent_cards[known_bid] = true;
        if (b) return true;
      }
    }
  }

  // player lost, opponent must have higher card
  if (wins[turn] == other_player(player)) {
    auto n = static_cast<int>(opponent_cards.size());
    for (int i = known_bid + 1; i < n; i++) {
      if (!opponent_cards[known_bid]) {
        continue;
      }
      else {
        opponent_cards[known_bid] = false;
        auto b =
            consistent(player,
                       opponent_cards,
                       turn + 1,
                       max_turn,
                       target_bids,
                       wins);
        opponent_cards[known_bid] = true;
        if (b) return true;
      }
    }
  }

  return false;
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
  const auto next_turn_n = current_bids.size();

  if (current_bids.size() < target_bids.size()) {
    if (current_game.player() == match_player) {
      const auto target = target_bids[next_turn_n];
      return { make_action(target) };
    }
    else {
      // NB we need to do some simple constraint solving here to find
      // infosets that are consistent with our observations
      auto cards = vector<bool>(target_game.n_cards(), false);
      for(auto card : current_game.hand(other_player(match_player))) {
        cards[card] = true;
      }

      auto s = set<action_t> { };
      for(auto card : current_game.hand(other_player(match_player))) {
        cards[card] = false;
        bool b = consistent(match_player,
                            cards,
                            current_game.turn(),
                            target_game.turn(),
                            target_bids, target_wins);
        if(b) {
          s.insert(make_action(card));
        }
        cards[card] = true;
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
