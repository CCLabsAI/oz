#include "hash.h"

#include "kuhn.h"

namespace oz {

using namespace std;

auto kuhn_poker_t::is_terminal() const -> bool {
  return showdown_ ||
      folded(player_t::P1) ||
      folded(player_t::P2);
}

void kuhn_poker_t::act_(action_t a) {
  if (player_ != CHANCE) {
    history_.push_back(a);
  }

  if (player_ == CHANCE) {
    deal_hand(a);
    player_ = P1;
  }
  else if (player_ == P1 && a == action_t::Pass) {
    if (pot(P2) > ANTE) {
      folded(P1) = true;
    }

    player_ = P2;
  }
  else if (player_ == P1 && a == action_t::Bet) {
    pot(P1) += 1;
    player_ = P2;
  }
  else if (player_ == P2 && a == action_t::Pass) {
    if (pot(P1) > ANTE) {
      folded(P2) = true;
    }
    else {
      showdown_ = true;
    }

    player_ = P1;
  }
  else if (player_ == P2 && a == action_t::Bet) {
    pot(P2) += 1;
    player_ = P1;
  }

  if (pot(P1) == 2 && pot(P2) == 2) {
    showdown_ = true;
  }
}

void kuhn_poker_t::deal_hand(action_t a) {
  if (!(a >= CHANCE_START && a <= CHANCE_FINISH)) {
    throw std::invalid_argument("illegal action");
  }

  switch(a) {
    case action_t::JQ:
      hand(P1) = card_t::Jack;
      hand(P2) = card_t::Queen;
      break;
    case action_t::JK:
      hand(P1) = card_t::Jack;
      hand(P2) = card_t::King;
      break;
    case action_t::QJ:
      hand(P1) = card_t::Queen;
      hand(P2) = card_t::Jack;
      break;
    case action_t::QK:
      hand(P1) = card_t::Queen;
      hand(P2) = card_t::King;
      break;
    case action_t::KJ:
      hand(P1) = card_t::King;
      hand(P2) = card_t::Jack;
      break;
    case action_t::KQ:
      hand(P1) = card_t::King;
      hand(P2) = card_t::Queen;
      break;

    default:
      assert (false);
  }
}

auto kuhn_poker_t::infoset() const -> oz::infoset_t {
  if (player_ == CHANCE) {
    return make_infoset<infoset_t>(player_, card_t::NA, history_);    
  }
  else {
    return make_infoset<infoset_t>(player_, hand(player_), history_);
  }
}

auto kuhn_poker_t::utility(player_t player) const -> value_t {
  assert (is_terminal());
  player_t winner;
  value_t u;

  if (showdown_) {
    winner = hand(P1) > hand(P2) ?
             P1 :
             P2;
  }
  else if (folded(P1)) {
    winner = P2;
  }
  else if (folded(P2)) {
    winner = P1;
  }
  else {
    assert (false);
    return 0;
  }

  if (winner == P1) {
    u = pot(P2);
  }
  else if (winner == P2) {
    u = -pot(P1);
  }
  else {
    assert (false);
    return 0;
  }

  return relative_utility(player, u);
}

auto kuhn_poker_t::infoset_t::actions() const -> vector<oz::action_t> {
  static const vector<oz::action_t> chance_actions {
    make_action(action_t::JQ),
    make_action(action_t::JK),
    make_action(action_t::QJ),
    make_action(action_t::QK),
    make_action(action_t::KJ),
    make_action(action_t::KQ)
  };

  static const vector<oz::action_t> player_actions {
    make_action(action_t::Pass),
    make_action(action_t::Bet),
  };

  if (player == CHANCE) {
    return chance_actions;
  } else {
    return player_actions;
  }
}

auto kuhn_poker_t::infoset_t::hash() const -> size_t {
    size_t seed = 0;
    ::hash_combine(seed, player);
    ::hash_combine(seed, hand);
    for (const auto &a : history) {
      ::hash_combine(seed, a);
    }
    return seed;
}

auto kuhn_poker_t::infoset_t::str() const -> string {
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

  if (!history.empty()) {
    ss << "/";
  }

  for (const auto& a : history) {
    if (a == action_t::Bet) {
      ss << "b";
    }
    else if (a == action_t::Pass) {
      ss << "p";
    }
    else { assert (false); }
  }

  return ss.str();
}

auto kuhn_poker_t::infoset_t::is_equal(const oz::infoset_t::concept_t &that)
const -> bool {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const kuhn_poker_t::infoset_t &>(that);
    return
      player == that_.player &&
      hand == that_.hand &&
      history == that_.history;
  }
  else {
    return false;
  }
}



} // namespace oz
