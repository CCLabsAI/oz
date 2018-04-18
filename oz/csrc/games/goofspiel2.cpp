#include "goofspiel2.h"

#include <sstream>

#include "hash.h"

namespace oz {

using namespace std;

goofspiel2_t::goofspiel2_t(int n_cards) :
  n_turns_(n_cards),
  turn_(0),
  player_(P1),
  score1_(0),
  score2_(0)
{
  for (int n = 0; n < n_cards; n++) {
    hand(P1).insert(n);
    hand(P2).insert(n);
  }
}

auto goofspiel2_t::infoset() const -> oz::infoset_t {
  Expects(player() != CHANCE);
  return make_infoset<infoset_t>(hand(player_), bids(player_), wins_);
}

auto goofspiel2_t::utility(player_t p) const -> value_t {
  Expects(is_terminal());
  value_t u;

  if(score(P1) > score(P2)) {
    u = 1;
  }
  else if (score(P1) < score(P2)) {
    u = -1;
  }
  else {
    u = 0;
  }

  return relative_utility(p, u);
}

template <typename T>
inline bool contains(const set<T> s, T x) {
  return s.find(x) != end(s);
}

void goofspiel2_t::act_(goofspiel2_t::action_t a) {
  card_t card = a;
  Expects(contains(hand(player_), card));

  if (player_ == P1) {
    P1_bid_ = card;
    hand(P1).erase(P1_bid_);

    player_ = P2;
  }
  else { // player_ == P2
    card_t P2_bid = card;
    hand(P2).erase(P2_bid);

    if (P2_bid < P1_bid_) {
      score(P1) += turn_;
      wins_.push_back(P1);
    }
    else if (P2_bid > P1_bid_) {
      score(P2) += turn_;
      wins_.push_back(P2);
    }
    else {
      // NB Chance player "winning" a bid means draw
      wins_.push_back(CHANCE);
    }

    bids(P1).push_back(P1_bid_);
    bids(P2).push_back(P2_bid);

    player_ = P1;
    turn_++;
  }
}

auto goofspiel2_t::chance_actions() const -> map<oz::action_t, prob_t> {
  Expects(false);
  return { };
}

auto goofspiel2_t::infoset_t::actions() const -> vector<oz::action_t> {
  return vector<oz::action_t>(begin(hand_), end(hand_));
}

auto goofspiel2_t::infoset_t::str() const -> string {
  stringstream ss;

  // TODO implement

  return ss.str();
}

bool goofspiel2_t::infoset_t::is_equal(const oz::infoset_t::concept_t& that)
const {
  if (typeid(*this) == typeid(that)) {
    auto that_ = static_cast<const goofspiel2_t::infoset_t&>(that);
    return bids_ == that_.bids_ &&
           wins_ == that_.wins_;
  }
  else {
    return false;
  }
}

size_t goofspiel2_t::infoset_t::hash() const {
  size_t seed = 0;
  for (const auto& bid : bids_) { hash_combine(seed, bid); }
  for (const auto& win : wins_) { hash_combine(seed, win); }
  return seed;
}


} // namespace oz
