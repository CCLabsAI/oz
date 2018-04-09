#ifndef OZ_ENCODER_H
#define OZ_ENCODER_H

#include <ATen/ATen.h>

#include "game.h"

namespace oz {

struct encoder_t {
  virtual int encoding_size() = 0;
  virtual void encode(infoset_t infoset, at::Tensor x) = 0;
};

class leduk_encoder_t : public encoder_t {
  static constexpr int ENCODING_SIZE = 6;

  int encoding_size() override { return ENCODING_SIZE; };
  void encode(infoset_t infoset, at::Tensor x) override;
};

};

#endif // OZ_ENCODER_H
