#include "history.h"

namespace oz {

using namespace std;

auto sample_chance(const history_t &history, rng_t& rng,
                   game_t::action_prob_allocator_t alloc) -> action_prob_t
{
  const auto actions_pr = history.chance_actions(alloc);
  Expects(!actions_pr.empty());

  auto actions = action_vector { };
  auto probs = prob_vector { };

  transform(begin(actions_pr), end(actions_pr), back_inserter(actions),
            [](const pair<action_t, prob_t> &x) -> action_t {
              return x.first;
            });

  transform(begin(actions_pr), end(actions_pr), back_inserter(probs),
            [](const pair<action_t, prob_t> &x) -> prob_t {
              return x.second;
            });

  auto a_dist = discrete_distribution<>(begin(probs), end(probs));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = probs[i];
  auto rho1 = pr_a;
  auto rho2 = pr_a;
  Ensures(0 <= pr_a && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

static auto sample_chance(const history_t &history, rng_t& rng)
-> action_prob_t {
  return sample_chance(history, rng, { });
}

auto history_t::sample_chance(oz::rng_t &rng) const -> action_prob_t {
  return oz::sample_chance(*this, rng);
}

static auto sample_uniform(const history_t &history, rng_t &rng)
-> action_prob_t
{
  auto actions = history.infoset().actions();
  auto N = static_cast<int>(actions.size());
  Expects(N > 0);

  auto d = uniform_int_distribution<>(0, N-1);
  auto i = d(rng);
  auto a = actions[i];
  auto pr_a = (prob_t) 1/N;
  Ensures(0 <= pr_a && pr_a <= 1);

  return { a, pr_a, pr_a, pr_a };
}

auto sample_action(const history_t &h, rng_t &rng) -> action_prob_t {
  if (h.player() == CHANCE) {
    return sample_chance(h, rng);
  }
  else {
    return sample_uniform(h, rng);
  }
}


} // namespace oz
