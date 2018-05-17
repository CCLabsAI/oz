#ifndef OZ_PY_SIGMA_BATCH
#define OZ_PY_SIGMA_BATCH

#include "sigma.h"
#include "oos.h"
#include "encoder.h"

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <unordered_map>

namespace oz {

class py_sigma_batch_t final {
 public:
  using lookup_t = std::unordered_map<infoset_t, action_prob_map_t>;

  void walk_infosets(history_t history);
  at::Tensor generate_batch(encoder_ptr_t encoder);
  void store_probs(encoder_ptr_t encoder, at::Tensor probs);

  class sigma_lookup_t final : public sigma_t::concept_t {
   public:
    sigma_lookup_t(lookup_t &lookup_table): lookup_table_(lookup_table) { }
    prob_t pr(infoset_t infoset, action_t a) const override;
   private:
    const lookup_t &lookup_table_;
  };

  sigma_t make_sigma();

 private:
  lookup_t lookup_table_;
};

} // namespace oz

#endif // OZ_PY_SIGMA_BATCH
