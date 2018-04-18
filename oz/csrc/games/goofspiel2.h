#ifndef OZ_GOOFSPIEL2_H
#define OZ_GOOFSPIEL2_H

#include <vector>
#include <set>

#include "game.h"

namespace oz {

using std::vector;
using std::set;

class goofspiel2_t final : public game_t {
 public:
  using card_t = int;
  using action_t = int;

  goofspiel2_t(int n_cards);
  goofspiel2_t(const goofspiel2_t &other) = default;

  class infoset_t : public oz::infoset_t::concept_t {
   public:
    explicit infoset_t(player_t player,
                       set<card_t> hand,
                       vector<card_t> bids,
                       vector<player_t> wins) :
      player_(player), hand_(hand), bids_(bids), wins_(wins) { };

    vector<oz::action_t> actions() const override;
    string str() const override;
    bool is_equal(const oz::infoset_t::concept_t& that) const override;
    size_t hash() const override;

   private:
    player_t player_;
    set<card_t> hand_;
    vector<card_t> bids_;
    vector<player_t> wins_;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(a.cast<action_t>()); };
  oz::infoset_t infoset() const override;
  player_t player() const override { return player_; }
  bool is_terminal() const override { return turn_ == n_turns_; }
  value_t utility(player_t player) const override;
  map<oz::action_t, prob_t> chance_actions() const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<goofspiel2_t>(*this);
  }

  set<card_t> &hand(player_t p) {
    switch (p) {
      case P1:
        return hand1_;
      case P2:
        return hand2_;
      default: assert (false);
        return hand1_;
    }
  }

  vector<card_t> &bids(player_t p) {
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

  const set<card_t> &hand(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->hand(p); }

  const vector<card_t> &bids(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->bids(p); }

  const int &score(player_t p) const
    { return const_cast<goofspiel2_t*>(this)->score(p); }

  vector<player_t> wins() { return wins_; }

 private:
  int n_turns_;
  int turn_;
  player_t player_;

  int score1_;
  int score2_;

  card_t P1_bid_;

  set<card_t> hand1_;
  set<card_t> hand2_;

  vector<card_t> bids1_;
  vector<card_t> bids2_;

  vector<player_t> wins_;
};

} // namespace oz

#endif // OZ_GOOFSPIEL2_H
