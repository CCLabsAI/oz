#ifndef OZ_LEDUC_H
#define OZ_LEDUC_H

#include "game.h"

#include <cassert>
#include <array>

#include <boost/container/static_vector.hpp>

namespace oz {

using std::move;
using std::string;
using std::array;
using std::vector;

using boost::container::static_vector;

class leduc_poker_t final : public game_t {
 public:

  enum class action_t {
    Raise = 1,
    Call,
    Fold,

    NextRound = -10,

    J1 = 100,
    Q1,
    K1,
    J2 = 200,
    Q2,
    K2,
    J = 1000,
    Q,
    K,
  };

  enum class card_t {
    NA = -1,
    Jack = 1,
    Queen,
    King,
  };

  static constexpr int N_PLAYERS = 2;
  static constexpr int MAX_ACTIONS = 10;
  using action_vector_t = static_vector<action_t, MAX_ACTIONS>;

  struct infoset_t : public oz::infoset_t::concept_t {
    const player_t player;
    const card_t hand;
    const card_t board;
    const action_vector_t history;
    const array<int, N_PLAYERS> pot;
    const int raises;

    infoset_t(player_t player, card_t hand, card_t board,
              action_vector_t history, array<int, N_PLAYERS> pot, int raises):
        player(player), hand(hand), board(board),
        history(move(history)), pot(pot), raises(raises) { }

    actions_list_t actions() const override;
    string str() const override;
    bool is_equal(const concept_t &that) const override;
    size_t hash() const override;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(a.cast<action_t>()); }
  oz::infoset_t infoset() const override;
  player_t player() const override { return player_; }
  bool is_terminal() const override;
  value_t utility(player_t player) const override;
  action_prob_map_t chance_actions() const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<leduc_poker_t>(*this);
  }

  oz::infoset_t infoset(oz::infoset_t::allocator_t alloc) const override;
  action_prob_map_t chance_actions(action_prob_allocator_t alloc) const override;

  std::string str() const override;

  static constexpr int ANTE = 1;
  static constexpr int N_ROUNDS = 2;
  static constexpr int RAISE_PER_ROUND[N_ROUNDS] = { 2, 4 };
  static constexpr int MAX_RAISES = 2;
  static constexpr int PAIR_RANK = 10;

 private:
  static constexpr action_t CHANCE_START = action_t::J1;
  static constexpr action_t CHANCE_FINISH = action_t::K;

  player_t player_ = CHANCE;
  array<card_t, N_PLAYERS> hand_ {{ card_t::NA, card_t::NA }};
  card_t board_ = card_t::NA;
  array<int, N_PLAYERS> pot_ {{ ANTE, ANTE }};
  int round_ = 0;
  bool checked_ = false;
  int raises_ = 0;
  action_vector_t history_;
  array<bool, N_PLAYERS> folded_ {{ false, false }};

  player_t other_player() const {
    Expects(player_ == P1 || player_ == P2);
    return player_ == P1 ? P2 : P1;
  }

  inline static int player_idx(player_t p) {
    Expects(p == P1 || p == P2);
    switch (p) {
      case P1: return 0;
      case P2: return 1;
      default: return 0; // NB not reachable
    }
  }

  void deal_hand(action_t a);
  void start_next_round();

  static int hand_rank(card_t card, card_t board);

 public:
  card_t hand(player_t p) const { return hand_[player_idx(p)]; }
  card_t &hand(player_t p) { return hand_[player_idx(p)]; }

  int pot(player_t p) const { return pot_[player_idx(p)]; }
  int &pot(player_t p) { return pot_[player_idx(p)]; }

  bool folded(player_t p) const { return folded_[player_idx(p)]; }
  bool &folded(player_t p) { return folded_[player_idx(p)]; }

  const action_vector_t &history() const { return history_; }

  int round() const { return round_; }
  card_t board() const { return board_; }
};

} // namespace oz

#endif // OZ_LEDUC_H
