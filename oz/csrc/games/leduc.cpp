#include "leduc.h"

#include "hash.h"

#include <cassert>
#include <memory>
#include <vector>
#include <sstream>

namespace oz {

using namespace std;

constexpr int leduc_poker_t::RAISE_PER_ROUND[];

void leduc_poker_t::act_(action_t a) {
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

void leduc_poker_t::start_next_round() {
  history_.push_back(action_t::NextRound);
  raises_ = 0;
  checked_ = false;
  round_ += 1;
  player_ = CHANCE;
}

void leduc_poker_t::deal_hand(action_t a) {
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

auto leduc_poker_t::is_terminal() const -> bool {
  return folded(P1) || folded(P2) || round_ >= N_ROUNDS;
}

auto leduc_poker_t::utility(player_t player) const -> value_t {
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

auto leduc_poker_t::hand_rank(card_t card, card_t board) -> int {
  if (card == board) {
    return PAIR_RANK;
  }
  else {
    return static_cast<int>(card);
  }
}

auto leduc_poker_t::infoset() const -> oz::infoset_t {
  Expects(player() != CHANCE);
  return make_infoset<infoset_t>(player_, hand(player_), board_,
                                 history_, pot_, raises_);
}

auto leduc_poker_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t
{
  Expects(player() != CHANCE);
  return allocate_infoset<infoset_t, oz::infoset_t::allocator_t>
               (alloc,
                player_, hand(player_), board_,
                history_, pot_, raises_);
}

static inline int card_idx(leduc_poker_t::card_t a) {
  using card_t = leduc_poker_t::card_t;

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

using count_vector_t = array<int, 3>;

static inline auto count_to_probs(const vector<oz::action_t> &actions,
                                  count_vector_t counts,
                                  leduc_poker_t::action_prob_allocator_t alloc)
  -> leduc_poker_t::action_prob_map_t
{
  Expects(actions.size() == counts.size());
  const prob_t total = accumulate(begin(counts), end(counts), (prob_t) 0);
  leduc_poker_t::action_prob_map_t m(alloc);

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

auto leduc_poker_t::chance_actions(action_prob_allocator_t alloc) const
  -> action_prob_map_t
{
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

  count_vector_t counts = { { 2, 2, 2 } };

  if (hand(P1) == card_t::NA) {
    return count_to_probs(chance_actions_p1, counts, alloc);
  }
  else if(hand(P2) == card_t::NA) {
    --counts[card_idx(hand(P1))];
    return count_to_probs(chance_actions_p2, counts, alloc);
  }
  else if(board_ == card_t::NA) {
    --counts[card_idx(hand(P1))];
    --counts[card_idx(hand(P2))];
    return count_to_probs(chance_actions_board, counts, alloc);
  }

  assert (false); // NB not reachable
  return { };
}

auto leduc_poker_t::chance_actions() const -> action_prob_map_t {
  return leduc_poker_t::chance_actions({ });
}

auto leduc_poker_t::infoset_t::actions() const -> actions_list_t {
  static const actions_list_t raise_call_fold {
    make_action(action_t::Raise),
    make_action(action_t::Call),
    make_action(action_t::Fold),
  };

  static const actions_list_t call_fold {
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

static std::ostream& print_card(std::ostream& os,
                                leduc_poker_t::card_t card,
                                const string &unk = "")
{
  using card_t = leduc_poker_t::card_t;

  if (card == card_t::Jack) {
    os << 'J';
  }
  else if (card == card_t::Queen) {
    os << 'Q';
  }
  else if (card == card_t::King) {
    os << 'K';
  }
  else {
    os << unk;
  }

  return os;
}

static std::ostream& operator <<(std::ostream& os,
                                 const leduc_poker_t::action_t &action)
{
  using action_t = leduc_poker_t::action_t;

  if (action == action_t::Raise) {
    os << 'r';
  }
  else if (action == action_t::Call) {
    os << 'c';
  }
  else if (action == action_t::Fold) {
    os << 'f';
  }
  else if (action == action_t::NextRound) {
    os << '/';
  }
  else {
    os << '?';
  }

  return os;
}


auto leduc_poker_t::infoset_t::str() const -> std::string {
  stringstream ss;

  print_card(ss, hand);
  print_card(ss, board);

  if (!history.empty()) {
    ss << "/";
  }

  for (const auto& a : history) {
    ss << a;
  }

  return ss.str();
}

auto leduc_poker_t::str() const -> std::string {
  stringstream ss;

  print_card(ss, hand(P1), "?");
  print_card(ss, hand(P2), "?");
  print_card(ss, board(), "?");

  if (!history().empty()) {
    ss << "/";
  }

  for (const auto& a : history()) {
    ss << a;
  }

  return ss.str();
}

bool leduc_poker_t::infoset_t::is_equal(const infoset_t::concept_t &that) const {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const leduc_poker_t::infoset_t &>(that);
    return
        hand    == that_.hand    &&
        board   == that_.board   &&
        history == that_.history;
  }
  else {
    return false;
  }
}

size_t leduc_poker_t::infoset_t::hash() const {
  size_t seed = 0;
  hash_combine(seed, hand);
  hash_combine(seed, board);
  for (const auto &a : history) { hash_combine(seed, a); }
  return seed;
}

} // namespace oz
