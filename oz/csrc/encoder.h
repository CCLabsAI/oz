#ifndef OZ_ENCODER_H
#define OZ_ENCODER_H

#include <ATen/ATen.h>

#include "game.h"
#include "oos.h"

#include "games/leduk.h"

namespace oz {

using std::map;
using at::Tensor;

class encoder_t {
 public:
  virtual int encoding_size() = 0;
  virtual int max_actions() = 0;
  virtual void encode(infoset_t infoset, Tensor x) = 0;
  virtual map<action_t, prob_t> decode(oz::infoset_t infoset, Tensor x) = 0;
  virtual action_prob_t decode_and_sample(infoset_t infoset, Tensor x,
                                          rng_t &rng) = 0;
};

} // namespace oz

#endif // OZ_ENCODER_H
