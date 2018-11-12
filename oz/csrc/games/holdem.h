#ifndef OZ_HOLDEM_H
#define OZ_HOLDEM_H

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

class holdem_poker_t final : public game_t {
 public:

  enum class action_t {
    Raise = 1,
    Call,
    Fold,

    NextRound = -10,
    Deal = 100
  };

  using card_t = int;
  static constexpr int CARD_NA = -1;

  static constexpr int N_PLAYERS = 2;
  static constexpr int MAX_ACTIONS = 10; // FIXME
  static constexpr int N_ROUNDS = 4;
  static constexpr int BIG_BLIND = 10;
  static constexpr int SMALL_BLIND = 5;
  static constexpr int RAISE_SIZE[N_ROUNDS] = { 10, 10, 20, 20 };
  static constexpr int FIRST_PLAYER[N_ROUNDS] = { 2, 1, 1, 1 };
  static constexpr int RAISE_PER_ROUND[N_ROUNDS] = { 2, 4 };
  static constexpr int MAX_RAISES[N_ROUNDS] = { 3, 4, 4, 4 };
  static constexpr int N_HOLE_CARDS = 2;
  static constexpr int MAX_BOARD_CARDS = 5;
  static constexpr int N_BOARD_CARDS[N_ROUNDS] = { 0, 3, 1, 1 };

  using hand_t = array<card_t, N_HOLE_CARDS>;
  using board_t = static_vector<card_t, MAX_BOARD_CARDS>;
  using action_vector_t = static_vector<action_t, MAX_ACTIONS>;

  struct infoset_t : public oz::infoset_t::concept_t {
    const player_t player;
    const hand_t hand;
    const board_t board;
    const action_vector_t history;
    const array<int, N_PLAYERS> pot;
    const int raises;

    infoset_t(player_t player, hand_t hand, board_t board,
              action_vector_t history, array<int, N_PLAYERS> pot, int raises):
        player(player), hand(hand), board(move(board)),
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
    return std::make_unique<holdem_poker_t>(*this);
  }

  oz::infoset_t infoset(oz::infoset_t::allocator_t alloc) const override;
  action_prob_map_t chance_actions(action_prob_allocator_t alloc) const override;

  std::string str() const override;

 private:
  player_t player_ = CHANCE;
  array<hand_t, N_PLAYERS> hand_ {{ {{CARD_NA, CARD_NA}}, {{CARD_NA, CARD_NA}} }};
  board_t board_ {{ CARD_NA, CARD_NA, CARD_NA, CARD_NA }};
  array<int, N_PLAYERS> pot_ {{ BIG_BLIND, SMALL_BLIND }};
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
      default: return 0; // should not be reachable
    }
  }

  void deal_hand(action_t a);
  void start_next_round();

  static int hand_rank(card_t card, card_t board);

 public:
  hand_t hand(player_t p) const { return hand_[player_idx(p)]; }
  hand_t &hand(player_t p) { return hand_[player_idx(p)]; }

  int pot(player_t p) const { return pot_[player_idx(p)]; }
  int &pot(player_t p) { return pot_[player_idx(p)]; }

  bool folded(player_t p) const { return folded_[player_idx(p)]; }
  bool &folded(player_t p) { return folded_[player_idx(p)]; }

  const action_vector_t &history() const { return history_; }

  int round() const { return round_; }
  const board_t &board() const { return board_; }
};

} // namespace oz

#endif // OZ_HOLDEM_H
