#ifndef OZ_HOLDEM_ENCODER_H
#define OZ_HOLDEM_ENCODER_H

#include <torch/torch.h>

#include "encoder.h"

#include "game.h"
#include "oos.h"

#include "games/holdem.h"

namespace oz {

using std::vector;
using std::map;

using at::Tensor;

class holdem_encoder_t final : public encoder_t {
 public:
  using card_t = holdem_poker_t::card_t;
  using action_t = holdem_poker_t::action_t;

  int encoding_size() override { return ENCODING_SIZE; };
  int max_actions() override { return MAX_ACTIONS; };
  void encode(oz::infoset_t infoset, Tensor x) override;
  void encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) override;
  map<oz::action_t, prob_t> decode(oz::infoset_t infoset, Tensor x) override;
  action_prob_t decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng) override;

 private:
  using nn_real_t = float;
  using ta_t = at::TensorAccessor<nn_real_t, 1>;

  static void card_one_hot(card_t card, ta_t &x_a, int i);
  static void action_one_hot(action_t action, ta_t &x_a, int i);
  static void rounds_one_hot(player_t player, const holdem_poker_t::action_vector_t &actions, ta_t &x_a, int i);

  static constexpr int N_PLAYERS = holdem_poker_t::N_PLAYERS;
  static constexpr int N_ROUNDS = holdem_poker_t::N_ROUNDS;
  static constexpr int N_RANKS = holdem_poker_t::N_RANKS;
  static constexpr int N_SUITS = holdem_poker_t::N_SUITS;
  static constexpr int N_HOLE_CARDS = holdem_poker_t::N_HOLE_CARDS;
  static constexpr int MAX_BOARD_CARDS = holdem_poker_t::MAX_BOARD_CARDS;

  static constexpr int CARD_SIZE = N_RANKS + N_SUITS;
  static constexpr int ACTION_SIZE = 2;
  static constexpr int MAX_ROUND_ACTIONS = 6; // 4 raises + 2 calls (crrrrc)
  static constexpr int MAX_CARDS = N_HOLE_CARDS + MAX_BOARD_CARDS;

  static constexpr int ENCODING_SIZE =
    MAX_CARDS*CARD_SIZE + N_PLAYERS*N_ROUNDS*MAX_ROUND_ACTIONS*ACTION_SIZE;
  static constexpr int MAX_ACTIONS = 3;
};

};

#endif // OZ_HOLDEM_ENCODER_H
