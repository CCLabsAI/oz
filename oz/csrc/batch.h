#ifndef OZ_BATCH_H
#define OZ_BATCH_H

#include "encoder.h"
#include "oos.h"
#include "tree.h"
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
                 prob_t eps, prob_t delta, prob_t gamma,
                 prob_t beta, prob_t eta);

  Tensor generate_batch();
  void step(Tensor probs, rng_t &rng);
  void step(rng_t &rng);

  void target(infoset_t target_infoset);

  const tree_t &tree() const { return tree_; }
  prob_t avg_targeting_ratio() const { return avg_targeting_ratio_; }

  void reset_targeting_ratio() { avg_targeting_ratio_ = 1.0; }

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
  prob_t beta_;
  prob_t eta_;

  prob_t avg_targeting_ratio_;
};

};

#endif // OZ_BATCH_H
