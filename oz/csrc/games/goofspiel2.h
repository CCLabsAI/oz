#ifndef OZ_GOOFSPIEL2_H
#define OZ_GOOFSPIEL2_H

#include "game.h"

#include <set>
#include <vector>
#include <bitset>

#include <boost/container/static_vector.hpp>

namespace oz {

using std::set;
using std::vector;
using std::bitset;

using boost::container::static_vector;

class goofspiel2_t final : public game_t {
 public:
  using card_t = uint8_t;
  using action_t = int;

  static constexpr int MAX_CARDS = 16;

  using hand_t = bitset<MAX_CARDS>;
  using bids_t = static_vector<card_t, MAX_CARDS>;
  using wins_t = static_vector<player_t, MAX_CARDS>;

  goofspiel2_t(int n_cards);
  goofspiel2_t(const goofspiel2_t &other) = default;

  class infoset_t : public oz::infoset_t::concept_t {
   public:
    explicit infoset_t(player_t player,
                       hand_t hand,
                       bids_t bids,
                       wins_t wins) :
      player_(player),
      hand_(move(hand)),
      bids_(move(bids)),
      wins_(move(wins)) { };

    actions_list_t actions() const override;
    string str() const override;
    bool is_equal(const oz::infoset_t::concept_t& that) const override;
    size_t hash() const override;

    player_t player() const { return player_; }
    const bids_t &bids() const { return bids_; }
    const wins_t &wins() const { return wins_; }

   private:
    player_t player_;
    hand_t hand_;
    bids_t bids_;
    wins_t wins_;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(a.cast<action_t>()); };
  oz::infoset_t infoset() const override;
  player_t player() const override { return player_; }
  bool is_terminal() const override { return turn_ == n_cards_; }
  value_t utility(player_t player) const override;
  action_prob_map_t chance_actions() const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<goofspiel2_t>(*this);
  }

  string str() const override;

  hand_t &hand(player_t p) {
    switch (p) {
      case P1:
        return hand1_;
      case P2:
        return hand2_;
      default: assert (false);
        return hand1_;
    }
  }

  bids_t &bids(player_t p) {
    switch (p) {
      case P1:
        return bids1_;
      case P2:
        return bids2_;
      default: assert (false);
        return bids1_;
    }
  }

  int &score(player_t p) {
    switch (p) {
      case P1:
        return score1_;
      case P2:
        return score2_;
      default: assert (false);
        return score1_;
    }
  }

  int n_cards() const { return n_cards_; }

  const hand_t &hand(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->hand(p); }

  const bids_t &bids(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->bids(p); }

  int score(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->score(p); }

  const wins_t &wins() const { return wins_; }

  int turn() const { return turn_; }

  // virtual ~goofspiel2_t() override;

 private:
  int n_cards_;
  int turn_;
  player_t player_;

  int score1_;
  int score2_;

  card_t P1_bid_;

  hand_t hand1_;
  hand_t hand2_;

  bids_t bids1_;
  bids_t bids2_;

  wins_t wins_;
};

} // namespace oz

#endif // OZ_GOOFSPIEL2_H
