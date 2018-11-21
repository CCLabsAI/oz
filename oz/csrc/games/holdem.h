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

namespace poker_cards {
  enum card_names {
    _2h, _3h, _4h, _5h, _6h, _7h, _8h, _9h, _Th, _Jh, _Qh, _Kh, _Ah,
    _2c, _3c, _4c, _5c, _6c, _7c, _8c, _9c, _Tc, _Jc, _Qc, _Kc, _Ac,
    _2d, _3d, _4d, _5d, _6d, _7d, _8d, _9d, _Td, _Jd, _Qd, _Kd, _Ad,
    _2s, _3s, _4s, _5s, _6s, _7s, _8s, _9s, _Ts, _Js, _Qs, _Ks, _As,
  };
}

class holdem_poker_t final : public game_t {
 public:
  using card_t = int;

  static constexpr int CARD_NA = -1;
  static constexpr card_t N_RANKS = 13;
  static constexpr card_t N_SUITS = 4;
  static constexpr card_t N_CARDS = 52;
  static constexpr card_t CARD_MIN = poker_cards::_2h;
  static constexpr card_t CARD_MAX = poker_cards::_As;

  static const std::string CARD_RANKS;
  static const std::string CARD_SUITS;

  static_assert(CARD_MAX-CARD_MIN == N_CARDS-1, "card index enum is incorrect size");

  static constexpr int DEAL_OFFSET = 10;

  enum class action_t {
    Raise = 1,
    Call,
    Fold,

    NextRound = -10,
    Deal = DEAL_OFFSET, // N.B. action = Deal + card_idx
    DealMax = DEAL_OFFSET + N_CARDS
  };

  static constexpr unsigned int N_PLAYERS = 2;
  static constexpr unsigned int MAX_ACTIONS = 28; // NB. ('crrrrc/') 7 * 4
  static constexpr unsigned int N_ROUNDS = 4;
  static constexpr unsigned int BIG_BLIND = 10;
  static constexpr unsigned int SMALL_BLIND = 5;
  static constexpr unsigned int RAISE_SIZE[N_ROUNDS] = { 10, 10, 20, 20 };
  static constexpr player_t FIRST_PLAYER[N_ROUNDS] = { P2, P1, P1, P1 };
  static constexpr unsigned int MAX_RAISES[N_ROUNDS] = { 3, 4, 4, 4 };
  static constexpr unsigned int N_HOLE_CARDS = 2;
  static constexpr unsigned int MAX_BOARD_CARDS = 5;
  static constexpr unsigned int N_BOARD_CARDS[N_ROUNDS] = { 0, 3, 4, 5 };

  enum class phase_t {
    DEAL_HOLE_P1,
    DEAL_HOLE_P2,
    DEAL_BOARD,
    BET,
    FINISHED
  };

  using hand_t = array<card_t, N_HOLE_CARDS>;
  using board_t = static_vector<card_t, MAX_BOARD_CARDS>;
  using action_vector_t = static_vector<action_t, MAX_ACTIONS>;


  struct infoset_t : public oz::infoset_t::concept_t {
    const player_t player;               // derived from history
    const hand_t hand;
    const board_t board;
    const action_vector_t history;
    const array<int, N_PLAYERS> pot;     // derived from history
    const bool can_raise;                // derived from history

    infoset_t(player_t player, hand_t hand, board_t board,
              action_vector_t history, array<int, N_PLAYERS> pot,
              bool can_raise):
        player(player), hand(hand), board(move(board)),
        history(move(history)), pot(pot), can_raise(can_raise) { }

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
  phase_t phase_ = phase_t::DEAL_HOLE_P1;
  player_t player_ = CHANCE;
  array<hand_t, N_PLAYERS> hand_ {{ {{CARD_NA, CARD_NA}}, {{CARD_NA, CARD_NA}} }};
  board_t board_;
  array<int, N_PLAYERS> pot_ {{ BIG_BLIND, SMALL_BLIND }};
  unsigned int round_ = 0;
  bool checked_ = false;
  unsigned int raises_ = 0;
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

  void dealer_act(action_t a);
  bool deal_hole_card(player_t player, card_t card);
  void start_next_round();

  bool can_raise() const;

 public:
  static bool is_deal_action(action_t a);
  static card_t card_for_deal_action(action_t action);
  static action_t deal_action_for_card(card_t card);

  static unsigned int hand_rank(const hand_t& hand, const board_t& board);

  hand_t hand(player_t p) const { return hand_[player_idx(p)]; }
  hand_t &hand(player_t p) { return hand_[player_idx(p)]; }

  int pot(player_t p) const { return pot_[player_idx(p)]; }
  int &pot(player_t p) { return pot_[player_idx(p)]; }

  bool folded(player_t p) const { return folded_[player_idx(p)]; }
  bool &folded(player_t p) { return folded_[player_idx(p)]; }

  const action_vector_t &history() const { return history_; }

  int round() const { return round_; }
  const board_t &board() const { return board_; }
  board_t &board() { return board_; }

  phase_t phase() const { return phase_; }
};

} // namespace oz

#endif // OZ_HOLDEM_H
