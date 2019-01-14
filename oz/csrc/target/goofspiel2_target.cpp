#include "oos.h"

#include "util.h"
#include "goofspiel2_target.h"

#include <bitset>

namespace oz {

using namespace std;
using card_t = goofspiel2_t::card_t;

static auto cast_history(const history_t &h) -> const goofspiel2_t& {
  return h.cast<goofspiel2_t>();
}

static auto cast_infoset(const infoset_t &infoset)
  -> const goofspiel2_t::infoset_t&
{
  return infoset.cast<goofspiel2_t::infoset_t>();
}

static auto other_player(player_t p) -> player_t {
  return p == P1 ? P2 : P1;
}

static constexpr int MAX_CARDS = std::numeric_limits<unsigned int>::digits;
using var_t = bitset<MAX_CARDS>;

static_assert(goofspiel2_t::MAX_CARDS <= MAX_CARDS, "MAX_CARDS not large enough");

static inline unsigned int unset_lsb(unsigned int x) {
  return x & (x-1);
}

static auto playable(const int turn,
                     const goofspiel2_t::hand_t &hand,
                     const int n_cards,
                     const player_t match_player,
                     const goofspiel2_t::bids_t &bids,
                     const goofspiel2_t::wins_t &wins) -> var_t
{
  Expects(bids.size() == wins.size());
  Expects(n_cards <= MAX_CARDS);

  // This takes elements of the AC-3 (the CSP one, not the ML one) algorithm
  // with the Leconte algorithm and the idea of hall sets
  // https://people.eng.unimelb.edu.au/pstuckey/papers/alldiff.pdf
  // https://en.wikipedia.org/wiki/AC-3_algorithm
  // It's fairly specialized to this problem.
  // NB. we only require "range consistency" because all
  // our constraints are essentially ranges.

  int n_vars = bids.size() - turn; // turns remaining
  var_t hand_bits(hand.to_ulong());
  var_t var[MAX_CARDS];

  // apply unit constraints
  for (int i = 0; i < n_vars; i++) {
    if (wins[turn + i] == CHANCE) {
      int n = bids[turn + i];
      var[i][n] = hand_bits[n];
    }
    else if (wins[turn + i] == match_player) {
      for (int n = 0; n < bids[turn + i]; n++) {
        var[i][n] = hand_bits[n];
      }
    }
    else if (wins[turn + i] == other_player(match_player)) {
      for (int n = bids[turn + i]+1; n < n_cards; n++) {
        var[i][n] = hand_bits[n];
      }
    }
  }

  // loop until stable
  bool changed;
  do {
    changed = false;

    // propagate vars with one value
    for (int i = 1; i < n_vars; i++) {
      for (int j = 0; j < i; j++) {
        if (var[j].count() == 1) {
          auto old_var = var[i];
          var[i] &= ~var[j];
          if (var[i] != old_var) changed = true;
        }

        if (var[i].count() == 1) {
          auto old_var = var[j];
          var[j] &= ~var[i];
          if (var[j] != old_var) changed = true;
        }
      }

      if (var[i].count() == 0) {
        return var_t { };
      }
    }

    // find and propagate hall sets
    // 1) loop through all ranges
    int a; unsigned int av = hand_bits.to_ulong();
    while ((a = __builtin_ffs(av))) {
      av = unset_lsb(av);

      int b; unsigned int bv = av;
      while ((b = __builtin_ffs(bv))) {
        bv = unset_lsb(bv);

        var_t range_bits;
        for (int n = 0; n < n_cards; n++) {
          if ((a-1) <= n && n <= (b-1)) range_bits[n] = hand_bits[n];
        }

        size_t range_size = range_bits.count();

        // 2) count the number of vars fully contained in this range
        size_t count_in_range = 0;
        for (int i = 0; i < n_vars; i++) {
          if ((var[i] & ~range_bits) == 0) count_in_range++;
        }

        // 3a) there are more vars than range elements: fail
        if (count_in_range > range_size) {
          return var_t { };
        }

        // 3b) there are exactly as many vars a range elements:
        // remove range from vars with any values outside this range
        if (count_in_range == range_size) {
          for (int i = 0; i < n_vars; i++) {
            if ((var[i] & ~range_bits) == 0) continue;

            auto old_var = var[i];
            var[i] &= ~range_bits;
            if (var[i] != old_var) changed = true;
          }
        }
      }
    }

  } while (changed);

  return var[0];
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
      var_t var = playable(next_turn,
                           opponent_hand, n_cards,
                           target_player, target_bids, target_wins);

      auto actions = set<action_t> { };

      for(int n = 0; n < n_cards; n++) {
        if(var[n]) actions.insert(make_action(n));
      }

      Ensures(!actions.empty());
      return actions;
    }
  }

  return { };
}

} // namespace oz
