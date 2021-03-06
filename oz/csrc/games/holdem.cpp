#include "holdem.h"

#include "hash.h"

#include <ace_eval.h>

#include <sstream>

namespace oz {

using namespace std;

constexpr int holdem_poker_t::CARD_NA;
constexpr unsigned int holdem_poker_t::RAISE_SIZE[];
constexpr player_t holdem_poker_t::FIRST_PLAYER[];
constexpr unsigned int holdem_poker_t::MAX_RAISES[];
constexpr unsigned int holdem_poker_t::N_BOARD_CARDS[];

const std::string holdem_poker_t::CARD_RANKS = "23456789TJQKA";
const std::string holdem_poker_t::CARD_SUITS = "hcds";


bool holdem_poker_t::is_deal_action(action_t a) {
  return action_t::Deal <= a && a < action_t::DealMax;
}

auto holdem_poker_t::card_for_deal_action(action_t action) -> card_t {
  auto card = static_cast<card_t>(static_cast<int>(action) - static_cast<int>(action_t::Deal));
  Ensures(CARD_MIN <= card && card <= CARD_MAX);
  return card;
}

auto holdem_poker_t::deal_action_for_card(card_t card) -> action_t {
  Expects(CARD_MIN <= card && card <= CARD_MAX);
  auto a = static_cast<action_t>(static_cast<int>(action_t::Deal) + static_cast<int>(card));
  return a;
}

bool holdem_poker_t::can_raise() const {
  return raises_ < MAX_RAISES[round_];
}

void holdem_poker_t::act_(action_t a) {
  if (player_ == CHANCE) {
    dealer_act(a);
  }

  else {
    Expects(phase_ == phase_t::BET);
    history_.push_back(a);

    if (a == action_t::Fold) {
      folded(player_) = true;
      player_ = CHANCE;
      phase_ = phase_t::FINISHED;
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
      if (raises_ > MAX_RAISES[round_]) {
        throw std::invalid_argument("illegal action: maximum raises reached");
      }

      int other_pot = pot(other_player());
      pot(player_) = other_pot + RAISE_SIZE[round_];
      raises_ += 1;
      player_ = other_player();
    }
    else {
      throw std::invalid_argument("illegal action: unknown action");
    }
  }
}

bool holdem_poker_t::deal_hole_card(player_t player, card_t card) {
  if (hand(player)[0] == CARD_NA) {
    hand(player)[0] = card;
    return false;
  }
  else if (hand(player)[1] == CARD_NA) {
    hand(player)[1] = card;
    return true;
  }
  else {
    throw std::invalid_argument("illegal action: tried to deal card to full hand");
  }
}

void holdem_poker_t::dealer_act(action_t a) {
  Expects(player() == CHANCE);

  if (!is_deal_action(a)) {
    throw std::invalid_argument("illegal action: not a deal action for chance player");
  }

  bool hand_full;
  auto card = card_for_deal_action(a);

  switch(phase_) {
    case phase_t::DEAL_HOLE_P1:
      hand_full = deal_hole_card(P1, card);
      if (hand_full) {
        phase_ = phase_t::DEAL_HOLE_P2;
      }
      break;

    case phase_t::DEAL_HOLE_P2:
      hand_full = deal_hole_card(P2, card);
      if (hand_full) {
        phase_ = phase_t::BET;
        player_ = FIRST_PLAYER[round_];
      }
      break;

    case phase_t::DEAL_BOARD:
      if (board().size() < N_BOARD_CARDS[round_]) {
        board().push_back(card);
      }
      else {
        throw std::invalid_argument("illegal action: tried to deal too many board cards");
      }

      if (board().size() == N_BOARD_CARDS[round_]) {
        phase_ = phase_t::BET;
        player_ = FIRST_PLAYER[round_];
      }

      break;

    default:
      throw std::invalid_argument("illegal action: tried to deal card outside dealing phase");
  }
}

void holdem_poker_t::start_next_round() {
  Expects(phase_ == phase_t::BET);

  history_.push_back(action_t::NextRound);
  raises_ = 0;
  checked_ = false;
  round_ += 1;
  player_ = CHANCE;

  if (round_ < N_ROUNDS) {
    phase_ = phase_t::DEAL_BOARD;
  } else {
    phase_ = phase_t::FINISHED;
  }
}

bool holdem_poker_t::is_terminal() const {
  return phase_ == phase_t::FINISHED;
}

unsigned int holdem_poker_t::hand_rank(const hand_t& hand, const board_t& board) {
  using namespace ace;

  Card h[ACEHAND] = { };

  if (hand[0] != CARD_NA) {
    ACE_addcard(h, ACE_makecard(hand[0]));
  }

  if (hand[1] != CARD_NA) {
    ACE_addcard(h, ACE_makecard(hand[1]));
  }

  for (holdem_poker_t::card_t c : board) {
    ACE_addcard(h, ACE_makecard(c));
  }

  return ACE_evaluate(h);
}

auto holdem_poker_t::utility(player_t player) const -> value_t {
  Expects(is_terminal());

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

using count_vector_t = array<int, holdem_poker_t::N_CARDS>;

static_assert(holdem_poker_t::N_HOLE_CARDS == 2, "current implementation assumes 2 hole cards");

static inline void decrement_count(int& total, count_vector_t& counts, holdem_poker_t::card_t card) {
  if (card != holdem_poker_t::CARD_NA) {
    --total;
    auto count = --counts[card];
    Expects(count >= 0);
  }
}

static inline void decrement_count(int& total, count_vector_t& counts, const holdem_poker_t::hand_t& hand) {
  decrement_count(total, counts, hand[0]);
  decrement_count(total, counts, hand[1]);
}

auto holdem_poker_t::chance_actions(action_prob_allocator_t alloc) const -> action_prob_map_t {
  Expects(player() == CHANCE);

  int total = N_CARDS;
  count_vector_t counts;

  std::fill(begin(counts), end(counts), 1);

  for (auto p : {P1, P2}) {
    decrement_count(total, counts, hand(p));
  }

  for (auto c : board()) {
    decrement_count(total, counts, c);
  }

  prob_t p = (prob_t) 1.0 / total;

  action_prob_map_t m(alloc);
  for (card_t c = CARD_MIN; c < N_CARDS; c++) {
    int n = counts[c];
    if (n > 0) {
      oz::action_t a = make_action(deal_action_for_card(c));
      prob_t pr_a = n * p;
      Ensures(0 <= pr_a && pr_a <= 1);
      m.emplace(a, pr_a);
    }
  }

  return m;
}

auto holdem_poker_t::chance_actions() const -> action_prob_map_t {
  return holdem_poker_t::chance_actions({ });
}

auto holdem_poker_t::infoset() const -> oz::infoset_t {
  if (player() == CHANCE) throw logic_error("cannot get infoset for chance player");

  return make_infoset<infoset_t>(player_, hand(player_), board_,
                                 history_, pot_, can_raise());
};

auto holdem_poker_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t
{
  if (player() == CHANCE) throw logic_error("cannot get infoset for chance player");

  return allocate_infoset<infoset_t, oz::infoset_t::allocator_t>
               (alloc,
                player_, hand(player_), board_,
                history_, pot_, can_raise());
}

// TODO prevent fold at start of round
auto holdem_poker_t::infoset_t::actions() const -> actions_list_t {
  static const actions_list_t raise_call_fold {
    make_action(action_t::Raise),
    make_action(action_t::Call),
    make_action(action_t::Fold),
  };

  static const actions_list_t call_fold {
    make_action(action_t::Call),
    make_action(action_t::Fold),
  };

  if (can_raise) {
    return raise_call_fold;
  }
  else {
    return call_fold;
  }
}


static std::ostream& print_card(std::ostream& os,
                                holdem_poker_t::card_t card)
{
  if (card == holdem_poker_t::CARD_NA) {
    os << "??";
    return os;
  }

  auto d = std::div(card, holdem_poker_t::N_RANKS);

  int rank = d.rem;
  int suit = d.quot;

  os << holdem_poker_t::CARD_RANKS[rank];
  os << holdem_poker_t::CARD_SUITS[suit];

  return os;
}

static std::ostream& print_hand(std::ostream& os,
                                const holdem_poker_t::hand_t& hand)
{
  print_card(os, hand[0]);
  print_card(os, hand[1]);

  return os;
}

static std::ostream& operator <<(std::ostream& os,
                                 const holdem_poker_t::action_t& action)
{
  using action_t = holdem_poker_t::action_t;

  switch (action) {
    case action_t::Raise:     os << 'r'; break;
    case action_t::Call:      os << 'c'; break;
    case action_t::Fold:      os << 'f'; break;
    case action_t::NextRound: os << '/'; break;
    default:                  os << '?'; break;
  }

  return os;
}

auto holdem_poker_t::str() const -> std::string {
  std::stringstream ss;

  print_hand(ss, hand(P1));
  ss << '|';
  print_hand(ss, hand(P2));

  if (!board().empty()) ss << '/';

  for (auto c : board()) {
    print_card(ss, c);
  }

  if (!history_.empty()) ss << ':';

  for (const auto& a : history_) {
    ss << a;
  }

  return ss.str();
};

auto holdem_poker_t::infoset_t::str() const -> std::string {
  std::stringstream ss;

  print_hand(ss, hand);

  if (!board.empty()) ss << '/';

  for (auto c : board) {
    print_card(ss, c);
  }

  if (!history.empty()) ss << ':';

  for (const auto& a : history) {
    ss << a;
  }

  return ss.str();
}

bool holdem_poker_t::infoset_t::is_equal(const concept_t &that) const {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const holdem_poker_t::infoset_t &>(that);
    return
        hand    == that_.hand    &&
        board   == that_.board   &&
        history == that_.history;
  }
  else {
    return false;
  }
}

size_t holdem_poker_t::infoset_t::hash() const {
  size_t seed = 0;
  hash_combine(seed, hand[0]);
  hash_combine(seed, hand[1]);
  for (const auto &c : board) { hash_combine(seed, c); }
  for (const auto &a : history) { hash_combine(seed, a); }
  return seed;
}

static holdem_poker_t::card_t card_for_rank_suit(int rank, int suit) {
  Expects(rank < holdem_poker_t::N_RANKS);
  Expects(suit < holdem_poker_t::N_SUITS);
  auto c = rank + holdem_poker_t::N_RANKS*suit;
  Ensures(holdem_poker_t::CARD_MIN <= c && c <= holdem_poker_t::CARD_MAX);
  return c;
}

static holdem_poker_t::card_t read_card(string s, size_t& pos) {
  char rank_char = std::toupper(s[pos++]);
  size_t rank = holdem_poker_t::CARD_RANKS.find_first_of(rank_char);
  if (rank == string::npos) {
    throw std::invalid_argument("invalid card rank");
  }

  char suit_char = std::tolower(s[pos++]);
  size_t suit = holdem_poker_t::CARD_SUITS.find_first_of(suit_char);
  if (suit == string::npos) {
    throw std::invalid_argument("invalid card suit");
  }

  return card_for_rank_suit(rank, suit);
}

static holdem_poker_t::action_t read_action(const string& s, size_t& pos) {
  auto a_char = s[pos++];
  switch (a_char) {
    case 'b':
    case 'r': return holdem_poker_t::action_t::Raise;
    case 'c': return holdem_poker_t::action_t::Call;
    case 'f': return holdem_poker_t::action_t::Fold;
    default:
      throw std::invalid_argument("invalid action");
  }
}

static void read_round(holdem_poker_t& g, const string& s, size_t& pos) {
  while (pos < s.size()) {
    if (s[pos] == '/') {
      pos++;
      break;
    }

    auto a = read_action(s, pos);
    g.act_(a);
  }
}

static const std::string UNKNOWN_HAND = "????";

static void read_hand(holdem_poker_t& g, const string& s, size_t& pos) {
  if (s.compare(pos, UNKNOWN_HAND.size(), UNKNOWN_HAND) == 0) {
    pos += UNKNOWN_HAND.size();
    return;
  }

  auto c1 = read_card(s, pos);
  g.act_(holdem_poker_t::deal_action_for_card(c1));

  auto c2 = read_card(s, pos);
  g.act_(holdem_poker_t::deal_action_for_card(c2));
}

void holdem_poker_t::read_history_str(string s) {
  if (!history_.empty()) {
    throw std::logic_error("cannot read history into game in progress");
  }

  size_t pos = 0;
  read_hand(*this, s, pos);

  if (s[pos] == '|') {
    pos++;
    read_hand(*this, s, pos);
  }
  else {
    if (pos == s.size()) {
      return;
    } else {
      throw std::invalid_argument("format error: expected P2 cards");
    }
  }

  board_t board;
  if (s[pos] == '/') {
    pos++;
    while (pos < s.size() && s[pos] != ':') {
      auto c = read_card(s, pos);
      board.push_back(c);
    }
  }

  auto board_it = begin(board);
  if (s[pos] == ':') {
    pos++;
    while (pos < s.size()) {
      read_round(*this, s, pos);

      while (phase() == phase_t::DEAL_BOARD && board_it != end(board)) {
        this->act_(deal_action_for_card(*board_it));
        ++board_it;
      }
    }
  }
  else {
    if (pos == s.size()) {
      return;
    } else {
      throw std::invalid_argument("format error: expected rounds");
    }
  }
}

} // namespace oz
