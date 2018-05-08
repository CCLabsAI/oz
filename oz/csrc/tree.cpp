#include "tree.h"
#include "history.h"

namespace oz {

using namespace std;

static auto sample_targeted(const node_t::node_sigma_t &sigma,
                            const infoset_t &infoset,
                            const node_t &node,
                            const std::set<action_t> &targets,
                            bool targeted,
                            prob_t eps,
                            prob_t gamma,
                            rng_t &rng) -> action_prob_t;

void tree_t::create_node(infoset_t infoset) {
  nodes_.emplace(infoset, node_t(infoset.actions()));
}

auto tree_t::sample_sigma(const infoset_t &infoset,
                          const set<action_t> &targets,
                          bool targeted,
                          bool average_response,
                          prob_t eps,
                          prob_t gamma,
                          rng_t &rng) const
  -> tree_t::sample_ret_t
{
  const auto it = nodes_.find(infoset);

  if (it == end(nodes_)) {
    return { { }, true };
  }
  else {
    const auto &node = lookup(infoset);
    const auto sigma = node.sigma_regret_matching();

    // const auto ap = sigma.sample_eps(infoset, eps, rng);
    const auto ap = sample_targeted(sigma, infoset, node,
                                    targets, targeted,
                                    eps, gamma,
                                    rng);

    return { ap, false };
  }
}

auto tree_t::sigma_average() const -> sigma_t {
  return make_sigma<sigma_average_t>(*this);
}

void tree_t::clear() {
  nodes_.clear();
}

auto sigma_average_t::pr(infoset_t infoset, action_t a) const -> prob_t {
  const auto actions = infoset.actions();
  const auto N = static_cast<int>(actions.size());

  const auto &nodes = tree_.nodes();
  const auto it = nodes.find(infoset);

  prob_t p;
  if (it != end(nodes)) {
    const auto &node = it->second;

    const auto total =
        accumulate(begin(actions), end(actions), (prob_t) 0,
                   [&node](const auto &x, const auto &a_prime) {
                     return x + node.average_strategy(a_prime);
                   });

    p = total > 0 ?
      node.average_strategy(a) / total :
      (prob_t) 1/N;
  }
  else {
    p = (prob_t) 1/N;
  }

  Ensures(0 <= p && p <= 1);
  return p;
};

// TODO break up this function
static auto sample_targeted(const node_t::node_sigma_t &sigma,
                            const infoset_t &infoset,
                            const node_t &node,
                            const set<action_t> &targets,
                            bool targeted,
                            prob_t eps,
                            prob_t gamma,
                            rng_t &rng) -> action_prob_t
{
  const auto actions = infoset.actions();

  Expects(!actions.empty());

  const auto N = static_cast<int>(actions.size());
  const auto p_eps = (prob_t) 1/N;
  const auto p_gamma = (prob_t) 1/N;

  // The difference between gamma and epsilon here is that
  // epsilon is extrinsic exploration that is accounted for by
  // importance weights (e.g. appears as a factor in rho)
  // gamma is considered part of the opponent strategy and models
  // a slightly fallible opponent that "slips" and plays a random
  // move with gamma probability.

  // raw action probabilities (with gamma "mistake" model)
  auto probs = prob_vector { };
  transform(begin(actions), end(actions), back_inserter(probs),
            [&](const action_t &a) -> prob_t {
              return gamma*p_gamma + (1 - gamma)*sigma.pr(infoset, a);
            });

  // epsilon exploration probabilities
  auto probs_untargeted = prob_vector { };
  transform(begin(probs), end(probs), back_inserter(probs_untargeted),
            [&](const prob_t &p) -> prob_t {
              return eps*p_eps + (1 - eps)*p;
            });

  // targeted action weights (unscaled)
  auto weights_targeted = prob_vector { };
  if (targets.empty()) {
    weights_targeted = probs_untargeted;
  }
  else {
    transform(begin(actions), end(actions),
              begin(probs_untargeted), back_inserter(weights_targeted),
              [&](const action_t &a, const prob_t &p) -> prob_t {
                if(targets.find(a) != end(targets)) {
                  return p;
                }
                else {
                  return 0;
                }
              });
  }

  const auto total_weight = accumulate(begin(weights_targeted),
                                       end(weights_targeted),
                                       (prob_t) 0);

  // NB if the targeted weights total is zero, we have tried to target a
  // subgame that has zero probability. In this case we bail out
  // and proceed by sampling in an untargeted way.
  // We also need to twiddle the rho1 probability in this case :/
  const auto &sample_probs = (total_weight > 0) ?
                             weights_targeted :
                             probs_untargeted;

  auto a_dist = discrete_distribution<>(begin(sample_probs),
                                        end(sample_probs));
  const auto i = a_dist(rng);

  const auto a = actions[i];
  const auto pr_a = probs[i];

  const auto pr_targeted   = weights_targeted[i] / total_weight;
  const auto pr_untargeted = probs_untargeted[i];

  const auto rho1 = (total_weight > 0) ? pr_targeted : pr_untargeted;
  const auto rho2 = pr_untargeted;

  // TODO does it makes sense for pr_a to be zero?
  Ensures(0 <= pr_a && pr_a <= 1);
  Ensures(0 <  rho1 && rho1 <= 1);
  Ensures(0 <  rho2 && rho2 <= 1);
  Ensures(!targeted || rho2 - rho1 < 1e-6);

  return { a, pr_a, rho1, rho2 };
}

} // namespace oz
