#include "sigma.h"

namespace oz {

using namespace std;

auto sigma_t::concept_t::sample_pr(infoset_t infoset, rng_t& rng) const
  -> action_prob_t
{
  auto actions = infoset.actions();
  auto probs = prob_vector(actions.size());

  transform(begin(actions), end(actions), begin(probs),
            [&](const auto& a) { return pr(infoset, a); });

  auto a_dist = discrete_distribution<>(begin(probs), end(probs));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = probs[i];
  auto rho1 = pr_a;
  auto rho2 = pr_a;

  return { a, pr_a, rho1, rho2 };
};

// TODO remove this function
auto sigma_t::sample_eps(infoset_t infoset, prob_t eps, rng_t &rng) const
  -> action_prob_t
{
  auto d_eps = uniform_real_distribution<>();
  const prob_t u = d_eps(rng);

  const auto actions = infoset.actions();
  Expects(!actions.empty());

  const auto N = static_cast<int>(actions.size());
  const auto p = (prob_t) 1/N;
  Expects(N > 0);

  if (u <= eps) {
    auto d_a = uniform_int_distribution<>(0, N-1);
    const auto i = d_a(rng);
    const auto a = actions[i];

    prob_t pr_a = pr(infoset, a);
    prob_t rho1 = eps*p + (1 - eps)*pr_a;
    prob_t rho2 = rho1;
    return { a, pr_a, rho1, rho2 };
  }
  else {
    auto ap = sample_pr(infoset, rng);
    ap.rho1 = eps*p + (1 - eps)*ap.pr_a;
    ap.rho2 = ap.rho1;
    return ap;
  }
};

} // namespace oz