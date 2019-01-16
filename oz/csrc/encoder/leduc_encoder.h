#ifndef OZ_LEDUC_ENCODER_H
#define OZ_LEDUC_ENCODER_H

#include <torch/torch.h>

#include "encoder.h"

#include "game.h"
#include "oos.h"

#include "games/leduc.h"

namespace oz {

using std::vector;
using std::map;

using at::Tensor;

class leduc_encoder_t final : public encoder_t {
 public:
  using card_t = leduc_poker_t::card_t;
  using action_t = leduc_poker_t::action_t;

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
  static void rounds_one_hot(const leduc_poker_t::action_vector_t &actions, ta_t &x_a, int i);

  static constexpr int N_ROUNDS = leduc_poker_t::N_ROUNDS;

  static constexpr int CARD_SIZE = 3;
  static constexpr int ACTION_SIZE = 2;
  static constexpr int ROUND_SIZE = 4*ACTION_SIZE;

  static constexpr int ENCODING_SIZE = 2*CARD_SIZE + N_ROUNDS*ROUND_SIZE;
  static constexpr int MAX_ACTIONS = 3;
};

};

#endif // OZ_LEDUC_ENCODER_H
