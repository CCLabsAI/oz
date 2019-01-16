#ifndef OZ_ENCODER_H
#define OZ_ENCODER_H

#include <torch/torch.h>

#include "game.h"
#include "sigma.h"

#include "games/leduc.h"

namespace oz {

using std::map;
using at::Tensor;

class encoder_t {
 public:
  virtual int encoding_size() = 0;
  virtual int max_actions() = 0;
  virtual void encode(infoset_t infoset, Tensor x) = 0;
  virtual void encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) = 0;
  virtual map<action_t, prob_t> decode(oz::infoset_t infoset, Tensor x) = 0;
  virtual action_prob_t decode_and_sample(infoset_t infoset, Tensor x,
                                          rng_t &rng) = 0;
};

using encoder_ptr_t = std::shared_ptr<encoder_t>;

} // namespace oz

#endif // OZ_ENCODER_H
