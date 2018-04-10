#ifndef OZ_LEDUK_H
#define OZ_LEDUK_H

#include <cassert>
#include <array>

#include "game.h"

namespace oz {

class leduk_poker_t : public game_t {

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

    // TODO clean this up
    DEAL1 = -2,
    DEAL2 = -3,
    DEAL_BOARD = -4
  };

  struct infoset_t : public oz::infoset_t::concept_t {
    const player_t player;
    const card_t hand;
    const card_t board;
    const std::vector<action_t> history;
    const std::array<int,2> pot;
    const int raises;

    infoset_t(player_t player, card_t hand, card_t board,
              std::vector<action_t> history, std::array<int,2> pot, int raises):
        player(player), hand(hand), board(board),
        history(std::move(history)), pot(pot), raises(raises) { }

    std::vector<oz::action_t> actions() const override;
    std::string str() const override;
    bool is_equal(const concept_t &that) const override;
    size_t hash() const override;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(static_cast<action_t>(a.index())); }
  oz::infoset_t infoset() const override; // FIXME
  player_t player() const override { return player_; }
  bool is_terminal() const override;
  value_t utility(player_t player) const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<leduk_poker_t>(*this);
  }

  static constexpr int N_PLAYERS = 2;
  static constexpr int ANTE = 1;
  static constexpr int N_ROUNDS = 2;
  static constexpr int RAISE_PER_ROUND[2] = { 2, 4 };
  static constexpr int MAX_RAISES = 2;
  static constexpr int PAIR_RANK = 10;

 private:
  static constexpr action_t CHANCE_START = action_t::J1;
  static constexpr action_t CHANCE_FINISH = action_t::K;

  player_t player_ = CHANCE;
  std::array<card_t, N_PLAYERS> hand_ { {card_t::NA, card_t::NA} };
  card_t board_ = card_t::NA;
  std::array<int, N_PLAYERS> pot_ { {ANTE, ANTE} };
  int round_ = 0;
  bool checked_ = false;
  int raises_ = 0;
  std::vector<action_t> history_;
  std::array<bool, N_PLAYERS> folded_ { {false, false} };

  player_t other_player() const {
    assert(player_ == P1 || player_ == P2);
    return player_ == P1 ? P2 : P1;
  }

  inline static int player_idx(player_t p) {
    switch (p) {
      case P1: return 0;
      case P2: return 1;
      default: throw std::invalid_argument("invalid player");
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

  int round() const { return round_; }
  card_t board() const { return board_; }
};

} // namespace oz

#endif // OZ_LEDUK_H
