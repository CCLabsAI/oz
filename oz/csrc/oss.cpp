#include <cassert>

#include "oss.h"

namespace oz {

void oss_t::search_t::step(action_prob_t ap) {
  assert (state_ == IN_TREE);
  assert (!history_.is_terminal());

  const auto acting_player = history_.player();
  const auto infoset = history_.infoset();

  if (acting_player == search_player_) {
    prefix_prob_.pi_i *= ap.pr_a;
  }
  else {
    prefix_prob_.pi_o *= ap.pr_a;
  }

  prefix_prob_.s1 *= ap.rho1;
  prefix_prob_.s2 *= ap.rho2;

  path_item_t path_item {
      acting_player, infoset,
      ap, prefix_prob_
  };

  history_.act(ap.a);
  path_.push_back(path_item);
}

void oss_t::search_t::walk(tree_t tree) {
  assert (state_ == IN_TREE);

  while (state_ == IN_TREE) {
    const auto acting_player = history_.player();

    if (history_.is_terminal()) {
      state_ = TERMINAL;
    }
    else if (acting_player == Chance) {
      action_prob_t ap = history_.sample_chance();
      step(ap);
    }
    else {
      infoset_t infoset = history_.infoset();
      action_prob_t ap {}; bool out_of_tree;
      std::tie(ap, out_of_tree) = tree.sample_sigma(infoset);

      if (out_of_tree) {
        state_ = PRIOR_EVAL;
      }
      else {
        step(ap);
      }
    }
  }
}

void oss_t::search_t::unwind(tree_t tree, suffix_prob_t prob) {
  suffix_prob_t suffix_prob = prob;

  for (auto i = path_.rbegin(); i != path_.rend(); ++i) {
    const auto& path_item = *i;
    const auto& active_player = path_item.player;
    const auto& infoset = path_item.infoset;
    const auto& action_prob = path_item.action_prob;
    const auto& prefix_prob = path_item.prefix_prob;

    action_t a = action_prob.a;

    prob_t pi_o = prefix_prob.pi_o;
    prob_t s1 = prefix_prob.s1;
    prob_t s2 = prefix_prob.s2;

    prob_t l = suffix_prob.l;
    prob_t c = suffix_prob.x;
    prob_t x = action_prob.pr_a * suffix_prob.x;
    value_t u = suffix_prob.u;

    node_t node = tree.lookup(infoset);

    if (active_player == search_player_) {
      value_t w = u*pi_o/l;
      for (const auto &a_prime : infoset.actions()) {
        value_t r;
        if (a_prime == a) {
          r = (c - x)*w;
        }
        else {
          r = -x*w;
        }

        node.accumulate_regret(a_prime, r);
      }
    }
    else {
      const auto sigma = node.sigma_regret_matching();

      prob_t q = delta_*s1 + (1 - delta_)*s2;
      for (const auto &a_prime : infoset.actions()) {
        value_t s = (1/q)*pi_o*sigma.pr(infoset, a_prime);
        node.accumulate_average_strategy(a_prime, s);
      }
    }

    suffix_prob.x = x;
  }
}

}
