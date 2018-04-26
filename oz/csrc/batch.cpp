#include <algorithm>
#include <iterator>
#include "batch.h"

namespace oz {

using namespace std;
using namespace at;

using search_list_t = batch_search_t::search_list_t;
using search_t = oos_t::search_t;

batch_search_t::batch_search_t(history_t root,
                               encoder_ptr_t encoder,
                               int batch_size) :
    root_(move(root)),
    encoder_(move(encoder)),
    batch_size_(batch_size)
{
  auto player = P1;

  generate_n(back_inserter(searches_), batch_size_, [&]() {
    const auto search_player = player;
    player = (player == P1 ? P2 : P1);
    return search_t { root_, search_player };
  });
}

bool search_needs_eval(const search_t &search) {
  using state_t = search_t::state_t;

  return (
    search.state() == state_t::PLAYOUT &&
    search.history().player() != CHANCE
  ) || (
    search.state() == state_t::CREATE
  );
}

static auto size0(const Tensor &x) {
  return x.dim() <= 0 ? 0 : x.size(0);
}

static auto count_needs_eval(const search_list_t &searches_) {
  return count_if(begin(searches_), end(searches_), search_needs_eval);
}

auto batch_search_t::generate_batch() -> Tensor {
  const auto N = static_cast<int>(searches_.size());
  const auto D = encoder_->encoding_size();
  Tensor d = CPU(kFloat).zeros({ N, D });

  int search_eval_n = 0;
  for (const auto &search : searches_) {
    if (search_needs_eval(search)) {
      int i = search_eval_n++;
      encoder_->encode(search.infoset(), d[i]);
    }
  }

  if (search_eval_n == 0) {
    return CPU(kFloat).zeros(0);
  }
  else {
    return d.narrow(0, 0, search_eval_n);
  }
}

void batch_search_t::step(Tensor avg, Tensor regret, rng_t &rng) {
  Expects(count_needs_eval(searches_) == size0(avg));
  Expects(size0(avg) == size0(regret));

  int search_eval_n = 0;
  for (auto &search : searches_) {
    const auto &history = search.history();
    switch (search.state()) {
      case oos_t::search_t::state_t::SELECT:
        search.select(tree_, rng);
        break;

      case oos_t::search_t::state_t::CREATE:
        {
          Expects(search_needs_eval(search));
          const int i = search_eval_n++;
          const auto infoset = search.infoset();
          const auto regrets = encoder_->decode(infoset, regret[i]);
          const auto average_strategy = encoder_->decode(infoset, avg[i]);

          const auto regrets_map = node_t::regret_map_t(begin(regrets),
                                                        end(regrets));

          const auto avg_map = node_t::avg_map_t(begin(average_strategy),
                                                 end(average_strategy));

          // search.create(tree_, rng);
          search.create_prior(tree_, regrets_map, avg_map, rng);
        }
        break;

      case oos_t::search_t::state_t::PLAYOUT:
        if (history.player() == CHANCE) {
          auto ap = history.sample_chance(rng);
          search.playout_step(ap);
        }
        else {
          Expects(search_needs_eval(search));
          const int i = search_eval_n++;
          const auto infoset = search.infoset();
          const auto ap = encoder_->decode_and_sample(infoset, avg[i], rng);
          search.playout_step(ap);
        }
        break;

      case oos_t::search_t::state_t::BACKPROP:
        search.backprop(tree_);
        break;

      case oos_t::search_t::state_t::FINISHED:
        // TODO is there a better way to do this?
        auto last_player = search.search_player();
        auto next_player = last_player == P1 ? P2 : P1;
        search = oos_t::search_t(root_, next_player);
        break;
    }
  }

  Expects(search_eval_n == size0(regret));
  Expects(search_eval_n == size0(avg));
}

} // namespace oz
