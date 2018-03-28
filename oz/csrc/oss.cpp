#include <cassert>
#include <iterator>

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

  path_item_t path_item{
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
    else if (acting_player == CHANCE) {
      action_prob_t ap = history_.sample_chance();
      step(ap);
    }
    else {
      infoset_t infoset = history_.infoset();
      action_prob_t ap{};
      bool out_of_tree;
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

void oss_t::search_t::unwind(tree_t tree, suffix_prob_t suffix_prob) {
  assert(state_ == TERMINAL);

  prob_t c;
  prob_t x = suffix_prob.x;

  const prob_t l = suffix_prob.l;
  const value_t u = suffix_prob.u;

  for (auto i = rbegin(path_); i != rend(path_); ++i) {
    const auto& path_item = *i;
    const auto& active_player = path_item.player;
    const auto& infoset = path_item.infoset;

    const auto a = path_item.action_prob.a;
    const auto pr_a = path_item.action_prob.pr_a;

    const auto pi_o = path_item.prefix_prob.pi_o;
    const auto s1 = path_item.prefix_prob.s1;
    const auto s2 = path_item.prefix_prob.s2;

    c = x;
    x = pr_a * x;

    node_t node = tree.lookup(infoset);

    if (active_player == search_player_) {
      const value_t w = u * pi_o / l;
      for (const auto& a_prime : infoset.actions()) {
        value_t r;
        if (a_prime == a) {
          r = (c - x) * w;
        }
        else {
          r = -x * w;
        }

        node.accumulate_regret(a_prime, r);
      }
    }
    else {
      const auto sigma = node.sigma_regret_matching();

      const prob_t q = delta_ * s1 + (1 - delta_) * s2;
      for (const auto& a_prime : infoset.actions()) {
        value_t s = (1 / q) * pi_o * sigma.pr(infoset, a_prime);
        node.accumulate_average_strategy(a_prime, s);
      }
    }
  }

  state_ = FINISHED;
}

auto sigma_t::concept_t::sample_pr(infoset_t infoset) const -> action_prob_t {
  auto actions = infoset.actions();
  auto prob_fn = [&](const action_t& a) { return pr(infoset, a); };
  std::vector<prob_t> probs(actions.size());
  std::transform(begin(actions), end(actions),
                 begin(probs), prob_fn);
  std::discrete_distribution<> d(begin(probs), end(probs));

  std::random_device rd;
  std::default_random_engine gen(rd());

  auto i = d(gen);

  auto a = actions[i];
  auto pr_a = probs[i];
  auto rho1 = pr_a;
  auto rho2 = pr_a;

  return { a, pr_a, rho1, rho2 };
};

sigma_t node_t::sigma_regret_matching() {
  assert (false);
}

void node_t::accumulate_regret(action_t a, value_t r) {
  assert (false);
}

void node_t::accumulate_average_strategy(action_t a, prob_t s) {
  assert (false);
}

action_prob_t history_t::sample_chance() {
  assert (false);
}

node_t tree_t::lookup(infoset_t infoset) {
  assert (false);
}

std::tuple<action_prob_t, bool> tree_t::sample_sigma(infoset_t infoset) {
  assert (false);
}

}
