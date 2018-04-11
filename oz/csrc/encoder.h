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
  int encoding_size() override { return ENCODING_SIZE; };
  int max_actions() override { return MAX_ACTIONS; };
  void encode(infoset_t infoset, at::Tensor x) override;
  action_prob_t decode_and_sample(infoset_t infoset, at::Tensor x, rng_t &rng) override;

 public:
  static constexpr int ACTION_SIZE = 2;
  static constexpr int ROUND_SIZE = 4*ACTION_SIZE;
  static constexpr int CARD_SIZE = 3;
  static constexpr int ENCODING_SIZE =
      2*CARD_SIZE + leduk_poker_t::N_ROUNDS*ROUND_SIZE;
  static constexpr int MAX_ACTIONS = 3;
};

};

#endif // OZ_ENCODER_H
