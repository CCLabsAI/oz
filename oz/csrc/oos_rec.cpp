#include "oos_rec.h"

/////////////////////////
// UNDER CONSTRUCTION  //
/////////////////////////

namespace oz { namespace rec {

template <class tree_t, class sigma_t>
auto oos_t<tree_t, sigma_t>::sample_action(history_t h, sigma_t sigma)
  -> action_prob_t
{
  if (h.player() == player_t::Chance) {
    action_prob2_t ret = sample_chance(move(h));
    return { ret.a, ret.s1 };
  }
  else {
    infoset_t infoset = h.infoset();
    return sigma.sample(infoset);
  }
}

template <class tree_t, class sigma_t>
auto oos_t<tree_t, sigma_t>::playout(history_t h, prob_t s, sigma_t sigma)
  -> walk_ret_t
{
  prob_t x = 1;

  while (!h.is_terminal()) {
    action_prob_t ap = sample_action(h, sigma);
    h.act(ap.a);
    x = x * ap.pr_a;
  }

  value_t u = h.utility(h.player());
  return { x, s * x, u };
}

template <class tree_t, class sigma_t>
auto oos_t<tree_t, sigma_t>::walk(history_t h,
                                  prob_t pi_i, prob_t pi_o,
                                  prob_t s1, prob_t s2, player_t i)
  -> walk_ret_t
{

  if (h.is_terminal()) {
    prob_t l = delta_ * s1 + (1 - delta_) * s2;
    value_t u = h.utility(h.player());
    return { 1, l, u };
  }

  else if (h.player() == player_t::Chance) {
    action_prob2_t ap = sample_chance(h);
    action_t a = ap.a;
    prob_t rho1 = ap.s1, rho2 = ap.s2;

    walk_ret_t r = walk(h >> a, pi_i, rho2 * pi_o, rho1 * s1, rho2 * s2, i);
    return { rho2 * r.x, r.l, r.u };
  }

  infoset_t infoset = h.infoset();

  typename tree_t::lookup_ret_t lr = tree_.lookup(infoset);
  typename tree_t::node_t node = lr.node;
  bool out_of_tree = lr.out_of_tree;

  sigma_t sigma;
  if (out_of_tree) {
    sigma = sigma_playout_;
  }
  else {
    sigma = node.sigma_regret_matching();
  }

  action_prob2_t ap = sample(move(h), sigma);
  action_t a = ap.a;
  prob_t s1_prime = ap.s1, s2_prime = ap.s2;

  prob_t pr_a = sigma.pr(infoset, a);

  prob_t x, l, u, c;
  if (out_of_tree) {
    prob_t q = delta_ * s1 + (1 - delta_) * s2;
    walk_ret_t ret = playout(h >> a, pr_a * q, sigma);
    x = ret.x, l = ret.l, u = ret.u;
  }
  else {
    prob_t pi_prime_i, pi_prime_o;
    if (h.player() == i) {
      pi_prime_i = pr_a * pi_i;
      pi_prime_o = pi_o;
    }
    else {
      pi_prime_i = pi_i;
      pi_prime_o = pr_a * pi_i;
    }

    walk_ret_t ret = walk(h >> a,
                          pi_prime_i, pi_prime_o,
                          s1_prime, s2_prime, i);
    x = ret.x, l = ret.l, u = ret.u;
  }

  c = x;
  x = pr_a * x;

  if (h.player() == i) {
    value_t w = u * pi_o / l;
    for (const auto& a_prime : infoset.actions()) {
      value_t r;
      if (a_prime == a) {
        r = (c - x) * w;
      }
      else {
        r = -x * w;
      }

      node.update_regret(a_prime, r);
    }
  }
  else {
    prob_t q = delta_ * s1 + (1 - delta_) * s2;
    for (const auto& a_prime : infoset.actions()) {
      value_t s = (1 / q) * pi_o * sigma.pr(infoset, a_prime);
      node.update_average_strategy(a_prime, s);
    }
  }

  return { x, l, u };
}

} } // namespace oz::rec
