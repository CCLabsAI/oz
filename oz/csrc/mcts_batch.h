#ifndef OZ_MCTS_BATCH_H
#define OZ_MCTS_BATCH_H

#include "mcts.h"
#include "encoder.h"

#include <ATen/ATen.h>

#include <vector>

namespace oz { namespace mcts {

using std::vector;
using at::Tensor;

class batch_search_t final {
 public:
  using search_t = oz::mcts::search_t;
  using encoder_ptr_t = std::shared_ptr<encoder_t>;
  using search_list_t = vector<oz::mcts::search_t>;

  batch_search_t(int batch_size,
                 history_t root,
                 encoder_ptr_t encoder,
                 params_t params);

  Tensor generate_batch();
  void step(Tensor probs, rng_t &rng);
  void step(rng_t &rng);

  const tree_t &tree() const { return tree_; }
  void target(infoset_t target_infoset);

 private:
  search_t make_search();

  int batch_size_;
  history_t root_;
  history_t target_;
  encoder_ptr_t encoder_;
  tree_t tree_;
  search_list_t searches_;

  params_t params_;
};

}}; // namespace oz::mcts

#endif // OZ_MCTS_BATCH_H
