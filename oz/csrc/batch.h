#ifndef OZ_BATCH_H
#define OZ_BATCH_H

#include <vector>

#include <ATen/ATen.h>

#include "encoder.h"
#include "oos.h"
#include "games/leduk.h"

namespace oz {

using std::vector;
using at::Tensor;

class batch_search_t {
 public:
  using encoder_ptr_t = std::shared_ptr<encoder_t>;
  using search_list_t = vector<oos_t::search_t>;

  batch_search_t(history_t root, encoder_ptr_t encoder, int batch_size);

  Tensor generate_batch();
  void step(Tensor d, rng_t &rng);

  const tree_t &tree() const { return tree_; }

 private:
  history_t root_;
  encoder_ptr_t encoder_;
  int batch_size_;
  tree_t tree_;
  search_list_t searches_;
};

};

#endif // OZ_BATCH_H
