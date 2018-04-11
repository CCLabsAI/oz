#ifndef OZ_ENCODER_H
#define OZ_ENCODER_H

#include <ATen/ATen.h>

#include "game.h"
#include "oos.h"

#include "games/leduk.h"

namespace oz {

class encoder_t {
 public:
  virtual int encoding_size() = 0;
  virtual int max_actions() = 0;
  virtual void encode(infoset_t infoset, at::Tensor x) = 0;
  virtual action_prob_t decode_and_sample(infoset_t infoset,
                                          at::Tensor x,
                                          rng_t &rng) = 0;
};

class leduk_encoder_t : public encoder_t {
 public:
  using nn_real_t = float;
  using ta_t = at::TensorAccessor<nn_real_t,1>;
  using game_encode_t = leduk_poker_t;
  using card_t = game_encode_t::card_t;
  using action_t = game_encode_t::action_t;

  int encoding_size() override { return ENCODING_SIZE; };
  int max_actions() override { return MAX_ACTIONS; };
  void encode(oz::infoset_t infoset, at::Tensor x) override;
  action_prob_t decode_and_sample(oz::infoset_t infoset, at::Tensor x, rng_t &rng) override;

 private:
  static const leduk_poker_t::infoset_t &cast_infoset(const infoset_t &infoset);
  static void card_one_hot(leduk_poker_t::card_t card, ta_t &x_a, int i);
  static void action_one_hot(leduk_poker_t::action_t action, ta_t &x_a, int i);
  static void rounds_one_hot(const std::vector<action_t> &actions, ta_t &x_a, int i);

  static constexpr int N_ROUNDS = leduk_poker_t::N_ROUNDS;

  static constexpr int CARD_SIZE = 3;
  static constexpr int ACTION_SIZE = 2;
  static constexpr int ROUND_SIZE = 4*ACTION_SIZE;

  static constexpr int ENCODING_SIZE =
      2*CARD_SIZE + leduk_poker_t::N_ROUNDS*ROUND_SIZE;
  static constexpr int MAX_ACTIONS = 3;
};

};

#endif // OZ_ENCODER_H
