#ifndef OZ_BATCH_H
#define OZ_BATCH_H

#include "encoder.h"
#include "oos.h"
#include "games/leduk.h"

#include <torch/torch.h>

#include <vector>

namespace oz {

using std::vector;
using at::Tensor;

class batch_search_t final {
 public:
  using encoder_ptr_t = std::shared_ptr<encoder_t>;
  using search_list_t = vector<oos_t::search_t>;

  batch_search_t(int batch_size,
                 history_t root,
                 encoder_ptr_t encode);

  batch_search_t(int batch_size,
                 history_t root,
                 encoder_ptr_t encoder,
                 target_t target,
                 prob_t eps, prob_t delta, prob_t gamma);

  Tensor generate_batch();
  void step(Tensor avg, Tensor regret, rng_t &rng);

  void retarget(infoset_t target_infoset);

  const tree_t &tree() const { return tree_; }

 private:
  oos_t::search_t make_search(player_t search_player);

  int batch_size_;
  history_t root_;
  encoder_ptr_t encoder_;
  tree_t tree_;
  search_list_t searches_;

  target_t target_;
  infoset_t target_infoset_;

  prob_t eps_;
  prob_t delta_;
  prob_t gamma_;
};

};

#endif // OZ_BATCH_H
