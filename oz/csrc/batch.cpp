#include <algorithm>
#include <iterator>
#include "batch.h"

namespace oz {

using namespace at;

batch_search_t::batch_search_t(history_t root,
                               batch_search_t::encoder_ptr_t encoder,
                               int batch_size) :
    root_(std::move(root)),
    encoder_(std::move(encoder)),
    batch_size_(batch_size)
{
  for (int i = 0; i < batch_size_; i++) {
    searches_.emplace_back(root_, P1);
  }
}

auto batch_search_t::generate_batch() -> Tensor {
  Tensor d = CPU(kFloat).ones({ batch_size_, encoder_->encoding_size() });

  for (size_t i = 0; i < searches_.size(); ++i) {
    encoder_->encode(searches_[i].infoset(), d[i]);
  }

  return d;
}

} // namespace oz