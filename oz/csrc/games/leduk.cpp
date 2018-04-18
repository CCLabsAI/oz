#include "leduk.h"

#include <cassert>
#include <memory>
#include <vector>
#include <sstream>

#include "hash.h"

namespace oz {

using namespace std;

constexpr int leduk_poker_t::RAISE_PER_ROUND[];

void leduk_poker_t::act_(action_t a) {
  if (player_ == CHANCE) {
    deal_hand(a);
  }

  else {
    history_.push_back(a);

    if (a == action_t::Fold) {
      folded(player()) = true;
    }
    else if (a == action_t::Call) {
      pot(player_) = pot(other_player());

      if (checked_ || raises_ > 0) {
        start_next_round();
      }
      else {
        checked_ = true;
        player_ = other_player();
      }
    }
    else if (a == action_t::Raise) {
      if (raises_ > MAX_RAISES) {
        throw std::invalid_argument("maximum raises reached");
      }

      int other_pot = pot(other_player());
      pot(player_) = other_pot + RAISE_PER_ROUND[round_];
      raises_ += 1;
      player_ = other_player();
    }
    else {
      throw std::invalid_argument("invalid action");
    }
  }
}

void leduk_poker_t::start_next_round() {
  history_.push_back(action_t::NextRound);
  raises_ = 0;
  checked_ = false;
  round_ += 1;
  player_ = CHANCE;
}

void leduk_poker_t::deal_hand(action_t a) {
  if (!(a >= CHANCE_START && a <= CHANCE_FINISH)) {
    throw std::invalid_argument("illegal action");
  }

  switch (a) {
    case action_t::J1:
      hand(P1) = card_t::Jack;
      break;
    case action_t::Q1:
      hand(P1) = card_t::Queen;
      break;
    case action_t::K1:
      hand(P1) = card_t::King;
      break;
    case action_t::J2:
      hand(P2) = card_t::Jack;
      break;
    case action_t::Q2:
      hand(P2) = card_t::Queen;
      break;
    case action_t::K2:
      hand(P2) = card_t::King;
      break;
    case action_t::J:
      board_ = card_t::Jack;
      break;
    case action_t::Q:
      board_ = card_t::Queen;
      break;
    case action_t::K:
      board_ = card_t::King;
      break;

    default: assert(false);
  }

  switch (a) {
    case action_t::J1:
    case action_t::Q1:
    case action_t::K1:
      player_ = CHANCE;
      break;

    case action_t::J2:
    case action_t::Q2:
    case action_t::K2:
      player_ = P1;
      break;

    case action_t::J:
    case action_t::Q:
    case action_t::K:
      player_ = P1;
      break;

    default: assert(false);
  }
}

auto leduk_poker_t::is_terminal() const -> bool {
  return folded(P1) || folded(P2) || round_ >= N_ROUNDS;
}

auto leduk_poker_t::utility(player_t player) const -> value_t {
  assert (is_terminal());

  value_t u;

  if (folded(P1)) {
    u = -pot(P1);
  }
  else if (folded(P2)) {
    u = pot(P2);
  }
  else {
    int p1_rank = hand_rank(hand(P1), board_);
    int p2_rank = hand_rank(hand(P2), board_);

    if (p1_rank == p2_rank) {
      u = 0;
    }
    else if (p1_rank > p2_rank) {
      u = pot(P2);
    }
    else {
      u = -pot(P1);
    }
  }

  return relative_utility(player, u);
}

auto leduk_poker_t::hand_rank(card_t card, card_t board) -> int {
  if (card == board) {
    return PAIR_RANK;
  }
  else {
    return static_cast<int>(card);
  }
}

auto leduk_poker_t::infoset() const -> oz::infoset_t {
  Expects(player() != CHANCE);
  return make_infoset<infoset_t>(player_, hand(player_), board_,
                                 history_, pot_, raises_);
}

static inline int card_idx(leduk_poker_t::card_t a) {
  using card_t = leduk_poker_t::card_t;

  switch(a) {
    case card_t::Jack:
      return 0;
    case card_t::Queen:
      return 1;
    case card_t::King:
      return 2;
    default: assert(false);
      return 0;
  }
}

static inline auto count_to_probs(vector<oz::action_t> actions,
                                  vector<int> counts)
  -> map<oz::action_t, prob_t>
{
  Expects(actions.size() == counts.size());
  const prob_t total = accumulate(begin(counts), end(counts), (prob_t) 0);
  auto m = map<oz::action_t, prob_t>();

  Expects(total > 0);

  // TODO cleanup loop
  auto count_it = begin(counts);
  for(auto action_it = begin(actions);
      action_it != end(actions);
      ++action_it, ++count_it) {
    int n = *count_it;
    if (n > 0) {
      prob_t p = (prob_t) *count_it / total;
      m.emplace(*action_it, p);
      Ensures(0 <= p && p <= 1);
    }
  }

  return m;
}

auto leduk_poker_t::chance_actions() const -> map<oz::action_t, prob_t> {
  Expects(player() == CHANCE);

  static const vector<oz::action_t> chance_actions_p1 {
      make_action(action_t::J1),
      make_action(action_t::Q1),
      make_action(action_t::K1),
  };

  static const vector<oz::action_t> chance_actions_p2 {
      make_action(action_t::J2),
      make_action(action_t::Q2),
      make_action(action_t::K2),
  };

  static const vector<oz::action_t> chance_actions_board {
      make_action(action_t::J),
      make_action(action_t::Q),
      make_action(action_t::K)
  };

  vector<int> counts = { 2, 2, 2 };

  if (hand(P1) == card_t::NA) {
    return count_to_probs(chance_actions_p1, counts);
  }
  else if(hand(P2) == card_t::NA) {
    --counts[card_idx(hand(P1))];
    return count_to_probs(chance_actions_p2, counts);
  }
  else if(board_ == card_t::NA) {
    --counts[card_idx(hand(P1))];
    --counts[card_idx(hand(P2))];
    return count_to_probs(chance_actions_board, counts);
  }

  assert (false);
  return map<oz::action_t, prob_t>();
}

auto leduk_poker_t::infoset_t::actions() const -> std::vector<oz::action_t> {
  static const vector<oz::action_t> raise_call_fold {
      make_action(action_t::Raise),
      make_action(action_t::Call),
      make_action(action_t::Fold),
  };

  static const vector<oz::action_t> call_fold {
      make_action(action_t::Call),
      make_action(action_t::Fold),
  };

  if (raises < MAX_RAISES) {
    return raise_call_fold;
  }
  else {
    return call_fold;
  }
}

auto leduk_poker_t::infoset_t::str() const -> std::string {
  stringstream ss;

  if (hand == card_t::Jack) {
    ss << "J";
  }
  else if (hand == card_t::Queen) {
    ss << "Q";
  }
  else if (hand == card_t::King) {
    ss << "K";
  }

  if (board == card_t::Jack) {
    ss << "J";
  }
  else if (board == card_t::Queen) {
    ss << "Q";
  }
  else if (board == card_t::King) {
    ss << "K";
  }

  if (!history.empty()) {
    ss << "/";
  }

  for (const auto& a : history) {
    if (a == action_t::Raise) {
      ss << "r";
    }
    else if (a == action_t::Call) {
      ss << "c";
    }
    else if (a == action_t::NextRound) {
      ss << "/";
    }
    else { assert (false); }
  }

  return ss.str();
}

bool leduk_poker_t::infoset_t::is_equal(const infoset_t::concept_t &that) const {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const leduk_poker_t::infoset_t &>(that);
    return
        player  == that_.player  &&
        hand    == that_.hand    &&
        board   == that_.board   &&
        history == that_.history &&
        pot     == that_.pot     &&
        raises  == that_.raises;
  }
  else {
    return false;
  }
}

size_t leduk_poker_t::infoset_t::hash() const {
  size_t seed = 0;
  hash_combine(seed, player);
  hash_combine(seed, hand);
  for (const auto &a : history) {
    hash_combine(seed, a);
  }
  hash_combine(seed, pot[0]);
  hash_combine(seed, pot[1]);
  hash_combine(seed, raises);
  return seed;
}

} // namespace oz
