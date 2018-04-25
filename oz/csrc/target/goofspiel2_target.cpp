#include "oos.h"

#include "util.h"
#include "goofspiel2_target.h"

namespace oz {

using namespace std;
using card_t = goofspiel2_t::card_t;

static auto cast_history(const history_t &h) -> const goofspiel2_t& {
  return h.cast<goofspiel2_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const goofspiel2_t::infoset_t& {
  return infoset.cast<goofspiel2_t::infoset_t>();
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

static constexpr size_t MAX_CARDS = 32;

auto bitset_to_set(bitset<MAX_CARDS> b) -> set<int> {
  auto s = set<int> { };

  for (int i = 0; i < b.size(); i++) {
    if (b[i]) {
      s.insert(i);
    }
  }

  return s;
}

static auto playable(const int turn,
                     const card_t card,
                     const set<card_t> &hand,
                     const int n_cards,
                     const player_t match_player,
                     const vector<card_t> &bids,
                     const vector<player_t> &wins) -> bool
{
  Expects(bids.size() == wins.size());
  Expects(n_cards <= MAX_CARDS);

  // This is basically the AC-3 algorithm
  // https://en.wikipedia.org/wiki/AC-3_algorithm
  // specialised to this problem. (the CSP one, not the ML one)

  int n_vars = bids.size() - turn; // turns remaining
  bitset<MAX_CARDS> hand_bits, work;
  bitset<MAX_CARDS> var[MAX_CARDS];

  for (const auto &hand_card : hand) {
    hand_bits[hand_card] = true;
  }

  for (int i = 0; i < n_vars; i++) {
    work[i] = true;
  }

  // apply unit constraints
  for (int i = 0; i < n_vars; i++) {
    if (wins[turn + i] == CHANCE) {
      int j = bids[turn + i];
      var[i][j] = hand_bits[j];
    }
    else if (wins[turn + i] == match_player) {
      for (int j = 0; j < bids[turn + i]; j++) {
        var[i][j] = hand_bits[j];
      }
    }
    else if (wins[turn + i] == other_player(match_player)) {
      for (int j = bids[turn + i]+1; j < n_cards; j++) {
        var[i][j] = hand_bits[j];
      }
    }
  }

  // assign card to be played this turn
  if(var[0][card]) {
    var[0].reset();
    var[0][card] = true;
  }
  else {
    return false;
  }


  // propagate constraints
//  while (work.any()) {
//    for (int i = 0; i < n_vars; i++) {
//      if(!work[i]) continue;
//      work[i] = false;
//
//      int n_vals = var[i].count();
//
//      if (n_vals == 0) {
//        return false;
//      }
//
//      if (n_vals == 1) {
//        for (int j = 0; j < n_vars; j++) {
//          if (i == j) continue;
//          auto old_var = var[j];
//          var[j] &= ~var[i];
//          bool changed = (var[j] != old_var);
//          work[j] = work[j] || changed;
//        }
//      }
//    }
//  }

  bool changed = true;
  while (changed) {
    changed = false;

    for (int i = 1; i < n_vars; i++) {
      for (int j = 0; j < i; j++) {
        if (var[j].count() == 1) {
          auto old_var = var[i];
          var[i] &= ~var[j];

          if (var[i] != old_var) {
            changed = true;
          }
        }

        if (var[i].count() == 1) {
          auto old_var = var[j];
          var[j] &= ~var[i];

          if (var[j] != old_var) {
            changed = true;
          }
        }
      }

      if (var[i].count() == 0) {
        return false;
      }
    }
  }

  return true;
}



auto goofspiel2_target_t::target_actions(const infoset_t &infoset,
                                         const history_t &current_history) const
  -> set<action_t>
{
  const auto &target_infoset = cast_infoset(infoset);
  const auto &target_player = target_infoset.player();

  const auto &current_game = cast_history(current_history);
  const auto &current_bids = current_game.bids(target_player);
  const auto &target_bids = target_infoset.bids();
  const auto &target_wins = target_infoset.wins();
  const auto n_cards = current_game.n_cards();

  Expects(target_bids.size() <= (unsigned) n_cards);

  const auto opponent = other_player(target_player);
  const auto &opponent_hand = current_game.hand(opponent);

  const auto next_turn = current_bids.size();

  if (current_bids.size() < target_bids.size()) {
    if (current_game.player() == target_player) {
      const auto target = target_bids[next_turn];
      return { make_action(target) };
    }
    else {
      const auto opponent_playable = [&](const card_t &card) -> bool {
        return playable(next_turn,
                        card, opponent_hand, n_cards,
                        target_player, target_bids, target_wins);
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

} // namespace oz
