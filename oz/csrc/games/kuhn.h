#ifndef OZ_KUHN_H
#define OZ_KUHN_H

#include <cassert>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>

#include "game.h"

namespace oz {

using std::move;
using std::string;
using std::vector;

class kuhn_poker_t final : public game_t {
 public:

  enum class action_t {
    Bet = 1,
    Pass,

    JQ = 100,
    JK,
    QJ,
    QK,
    KJ,
    KQ
  };

  enum class card_t {
    NA = -1,
    Jack = 1,
    Queen,
    King,
  };

  struct infoset_t : public oz::infoset_t::concept_t {
    const player_t player;
    const card_t hand;
    const vector<action_t> history;

    infoset_t(player_t player, card_t hand, vector<action_t> history):
      player(player), hand(hand), history(move(history)) { }

    vector<oz::action_t> actions() const override;
    string str() const override;
    bool is_equal(const concept_t &that) const override;
    size_t hash() const override;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(a.cast<action_t>()); };
  oz::infoset_t infoset() const override;
  player_t player() const override { return player_; }
  bool is_terminal() const override;
  value_t utility(player_t player) const override;
  map<oz::action_t, prob_t> chance_actions() const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<kuhn_poker_t>(*this);
  }

 private:
  static constexpr int N_PLAYERS = 2;
  static constexpr int ANTE = 1;
  static constexpr action_t CHANCE_START = action_t::JQ;
  static constexpr action_t CHANCE_FINISH = action_t::KQ;

  bool showdown_ = false;
  bool folded_[N_PLAYERS] = {false, false};
  card_t hand_[N_PLAYERS] = {card_t::NA, card_t::NA};
  int pot_[N_PLAYERS] = {ANTE, ANTE};
  player_t player_ = player_t::Chance;
  vector<action_t> history_;

  void deal_hand(action_t a);

  static int player_idx(player_t p) {
    switch (p) {
      case P1: return 0;
      case P2: return 1;
      default: throw std::invalid_argument("invalid player");
    }
  }

 public:
  card_t hand(player_t p) const { return hand_[player_idx(p)]; }
  card_t &hand(player_t p) { return hand_[player_idx(p)]; }

  int pot(player_t p) const { return pot_[player_idx(p)]; }
  int &pot(player_t p) { return pot_[player_idx(p)]; }

  bool folded(player_t p) const { return folded_[player_idx(p)]; }
  bool &folded(player_t p) { return folded_[player_idx(p)]; }
};

} // namespace oz

#endif // OZ_KUHN_H
