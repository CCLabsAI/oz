#include "py_sigma_batch.h"

#include "best_response.h"

#include <iterator>

namespace oz {

using namespace std;
using namespace at;

void py_sigma_batch_t::walk_infosets(history_t history) {
  if (history.is_terminal()) {
    return;
  }

  if (history.player() != CHANCE) {
    const auto infoset = history.infoset();
    lookup_table_[infoset];
  }

  for (const auto& a : actions(history)) {
    walk_infosets(history >> a);
  }
}

Tensor py_sigma_batch_t::generate_batch(encoder_ptr_t encoder) {
  const long N = lookup_table_.size();
  const long D = encoder->encoding_size();
  Tensor t = zeros(torch::CPU(kFloat), { N, D });

  int i = 0;
  for (const auto &p : lookup_table_) {
    const auto &infoset = p.first;
    encoder->encode(infoset, t[i++]);
  }

  Ensures(i == t.size(0));
  return t;
}

void py_sigma_batch_t::store_probs(encoder_ptr_t encoder, Tensor probs) {
  int i = 0;

  for (auto &p : lookup_table_) {
    const auto &infoset = p.first;
    auto &action_probs = p.second;

    const auto decoded = encoder->decode(infoset, probs[i++]);

    const auto total = accumulate(begin(decoded), end(decoded), (prob_t) 0.0,
      [](const auto &r, const auto &p) { return r + p.second; });

    const auto pr_uniform = (prob_t) 1.0 / decoded.size();

    action_probs.reserve(decoded.size());
    action_probs.clear();

    std::copy(begin(decoded), end(decoded),
              inserter(action_probs, end(action_probs)));

    if (total > 0) {
      for_each(begin(action_probs), end(action_probs),
        [&](auto &p) { p.second = p.second / total; });
    }
    else {
      for_each(begin(action_probs), end(action_probs),
        [&](auto &p) { p.second = pr_uniform; });
    }
  }

  Ensures(i == probs.size(0));
}

sigma_t py_sigma_batch_t::make_sigma() {
  return oz::make_sigma<py_sigma_batch_t::sigma_lookup_t>(lookup_table_);
}

prob_t py_sigma_batch_t::sigma_lookup_t::pr(infoset_t infoset, action_t a) const {
  const auto &action_probs = lookup_table_.at(infoset);
  const prob_t pr_a = action_probs.at(a);

  return pr_a;
}

} // namespace oz
