#include "node.h"

namespace oz {

using namespace std;

node_t::node_t(infoset_t::actions_list_t actions) {
  Expects(!actions.empty());

  static const auto make_zero_value =
      [](const action_t &a) { return make_pair(a, 0); };

  regrets_.reserve(actions.size());
  average_strategy_.reserve(actions.size());
  prior_.reserve(actions.size());

  transform(begin(actions), end(actions),
            inserter(regrets_, end(regrets_)),
            make_zero_value);

  transform(begin(actions), end(actions),
            inserter(average_strategy_, end(average_strategy_)),
            make_zero_value);

  transform(begin(actions), end(actions),
            inserter(prior_, end(prior_)),
            make_zero_value);

  Ensures(!regrets_.empty());
  Ensures(!average_strategy_.empty());
  Ensures(!prior_.empty());
}

sigma_regret_prior_t node_t::sigma_regret_matching() const {
  auto alpha = (prob_t) 1.0 / (regret_n() + 1);
  return sigma_regret_prior_t(regrets_,
                              prior_,
                              alpha);
}

auto sigma_regret_t::pr(infoset_t infoset, action_t a) const -> prob_t {
  auto sum_positive = accumulate(
    begin(regrets_), end(regrets_), (value_t) 0,
    [](const auto &r, const auto &x) {
        return r + max<value_t>(0, x.second);
    });

  prob_t p;
  if (sum_positive > 0) {
    auto r = regrets_.at(a);
    p = r > 0 ? r/sum_positive : 0;
  }
  else {
    p = (prob_t) 1/infoset.actions().size();
  }

  Ensures(0 <= p && p <= 1);
  return p;
}

auto sigma_regret_t::sample_pr(infoset_t infoset, rng_t &rng) const
  -> action_prob_t
{
  auto actions = action_vector { };
  auto weights = prob_vector { };

  transform(begin(regrets_), end(regrets_), back_inserter(actions),
            [](const auto &x) { return x.first; });

  transform(begin(regrets_), end(regrets_), back_inserter(weights),
            [](const auto &x) { return max<value_t>(0, x.second); });

  auto total = accumulate(begin(weights), end(weights), (prob_t) 0);
  auto N = static_cast<int>(weights.size());

  Expects(N > 0);
  Expects(!actions.empty());
  Expects(!weights.empty());
  Expects(actions.size() == weights.size());

  int i;
  if (total > 0) {
    auto a_dist = discrete_distribution<>(begin(weights), end(weights));
    i = a_dist(rng);
  }
  else {
    auto a_dist = uniform_int_distribution<>(0, N-1);
    i = a_dist(rng);
  }

  auto a = actions[i];
  auto pr_a = total > 0 ? weights[i]/total : (prob_t) 1/N;
  auto rho1 = pr_a, rho2 = pr_a;

  Ensures(0 <= pr_a && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

prob_t sigma_regret_prior_t::pr(infoset_t infoset, action_t a) const {
  auto sum_positive = accumulate(
      begin(regrets_), end(regrets_), (value_t) 0,
      [&](const auto &r, const auto &x) {
        return r + max<value_t>(0, x.second);
      });

  prob_t p;
  if (sum_positive > 0) {
    auto r = regrets_.at(a);
    p = r > 0 ? r/sum_positive : 0;
  }
  else {
    p = (prob_t) 1/infoset.actions().size();
  }

  auto alpha = prior_alpha_;
  p = alpha*prior_.at(a) + (1 - alpha)*p;

  Ensures(0 <= p && p <= 1);
  return p;
}

} // namespace oz
