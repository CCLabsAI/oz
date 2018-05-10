#include "mcts_batch.h"

namespace oz { namespace mcts {

using namespace std;
using namespace at;

using search_t = batch_search_t::search_t;
using search_list_t = batch_search_t::search_list_t;

auto batch_search_t::make_search() -> search_t {
  return search_t { target_, params_ };
}

batch_search_t::batch_search_t(int batch_size,
                               history_t root,
                               encoder_ptr_t encoder,
                               params_t params) :
  batch_size_(batch_size),
  root_(move(root)),
  target_(root_),
  encoder_(move(encoder)),
  params_(params)
{
  generate_n(back_inserter(searches_), batch_size_, [&]() {
    return this->make_search();
  });
}

// TODO remove duplication

static bool search_needs_eval(const search_t &search) {
  using state_t = search_t::state_t;

  return (
      search.state() == state_t::PLAYOUT &&
          search.history().player() != CHANCE
  ) || (
      search.state() == state_t::CREATE
  );
}

static auto count_needs_eval(const search_list_t &searches_) {
  return count_if(begin(searches_), end(searches_), search_needs_eval);
}

auto batch_search_t::generate_batch() -> Tensor {
  const auto N = count_needs_eval(searches_);
  const auto D = encoder_->encoding_size();
  Tensor d = zeros(torch::CPU(kFloat), { N, D });

  int i = 0;
  for (const auto &search : searches_) {
    if (search_needs_eval(search)) {
      encoder_->encode(search.infoset(), d[i++]);
    }
  }

  Ensures(i == N);

  return d;
}

void batch_search_t::step(Tensor probs, rng_t &rng) {
  using state_t = search_t::state_t;

  auto N = count_needs_eval(searches_);
  Expects(N == probs.size(0));

  int i = 0;
  for (auto &search : searches_) {
    const auto &history = search.history();
    switch (search.state()) {
      case state_t::SELECT:
        search.select(tree_, rng);
        break;

      case state_t::CREATE:
      {
        Expects(search_needs_eval(search));
        const int n = i++;
        const auto infoset = search.infoset();

        const auto average_strategy = encoder_->decode(infoset, probs[n]);

        const auto avg_map = action_prob_map_t(begin(average_strategy),
                                               end(average_strategy));

        search.create(tree_, rng);
//        search.create_prior(tree_, avg_map, rng);
      }
        break;

      case state_t::PLAYOUT:
        if (history.player() == CHANCE) {
          auto ap = history.sample_chance(rng);
          search.playout_step(ap.a);
        }
        else {
          Expects(search_needs_eval(search));
          const int n = i++;
          const auto infoset = search.infoset();
          const auto ap = encoder_->decode_and_sample(infoset, probs[n], rng);
          search.playout_step(ap.a);
        }
        break;

      case state_t::BACKPROP:
        search.backprop(tree_);
        break;

      case state_t::FINISHED:
        search = make_search();
        break;
    }
  }

  Ensures(i == N);
}

void batch_search_t::step(rng_t &rng) {
  step(torch::CPU(kFloat).tensor(), rng);
}

}}; // namespace oz::mcts
