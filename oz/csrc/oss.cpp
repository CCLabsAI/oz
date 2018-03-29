#include <cassert>
#include <algorithm>
#include <iterator>

#include <unordered_map>

#include "oss.h"

namespace oz {

using namespace std;

void update_probs(oss_t::prefix_prob_t& probs, player_t i,
  const action_prob_t ap, player_t p) {
  if(p == i) {
    probs.pi_i *= ap.pr_a;
  }
  else {
    probs.pi_o *= ap.pr_a;
  }

  probs.s1 *= ap.rho1;
  probs.s2 *= ap.rho2;
}

void oss_t::search_t::step_tree(action_prob_t ap) {
  assert (state_ == state_t::SELECT);
  assert (!history_.is_terminal());

  const auto acting_player = history_.player();
  const auto infoset = history_.infoset();

  // save the current infoset and prefix stats
  path_.emplace_back(path_item_t {
      acting_player, infoset,
      ap, prefix_prob_
  });

  // update state and sample probabilities
  history_.act(ap.a);
  update_probs(prefix_prob_, search_player_, ap, acting_player);
}

void oss_t::search_t::select(tree_t& tree) {
  assert (state_ == state_t::SELECT);

  while (state_ == state_t::SELECT) {
    const auto acting_player = history_.player();

    if (history_.is_terminal()) {
      state_ = state_t::BACKPROP;
    }
    else if (acting_player == CHANCE) {
      auto ap = history_.sample_chance();
      step_tree(ap);
    }
    else {
      infoset_t infoset = history_.infoset();
      const auto r = tree.sample_sigma(infoset);

      if (r.out_of_tree) {
        state_ = state_t::CREATE;
      }
      else {
        step_tree(r.ap);
      }
    }
  }
}

void oss_t::search_t::create(tree_t& tree) {
  assert (state_ == state_t::CREATE);
  assert (!history_.is_terminal());

  const auto infoset = history_.infoset();
  tree.create_node(infoset);
  
  const auto r = tree.sample_sigma(infoset);
  assert (!r.out_of_tree);
  
  step_tree(r.ap);

  if (history_.is_terminal()) {
    state_ = state_t::BACKPROP;
  }
  else {
    state_ = state_t::PLAYOUT;    
  }
}

void oss_t::search_t::playout_step(action_prob_t ap) {
  assert (state_ == state_t::PLAYOUT);
  
  history_.act(ap.a);
  suffix_prob_.x *= ap.pr_a;

  if (history_.is_terminal()) {
    prob_t s1 = prefix_prob_.s1;
    prob_t s2 = prefix_prob_.s2;

    suffix_prob_.l = delta_ * s1 + (1.0 - delta_) * s2;
    suffix_prob_.u = history_.utility(search_player_);
    
    state_ = state_t::BACKPROP;
  }
}

void oss_t::search_t::backprop(tree_t& tree) {
  assert (state_ == state_t::BACKPROP);
  assert (history_.is_terminal());

  prob_t c = 1.0;
  prob_t x = suffix_prob_.x;

  const prob_t  l = suffix_prob_.l;
  const value_t u = suffix_prob_.u;

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

      const prob_t q = delta_ * s1 + (1.0 - delta_) * s2;
      for (const auto& a_prime : infoset.actions()) {
        prob_t s = (pi_o * sigma.pr(infoset, a_prime)) / q;
        node.accumulate_average_strategy(a_prime, s);
      }
    }
  }

  state_ = state_t::FINISHED;
}

auto sigma_t::concept_t::sample_pr(infoset_t infoset, rng_t& rng) const -> action_prob_t {
  auto actions = infoset.actions();
  auto probs = vector<prob_t>(actions.size());
  
  transform(begin(actions), end(actions), begin(probs),
            [&](auto& a) { return pr(infoset, a); });
  
  auto a_dist = discrete_distribution<>(begin(probs), end(probs));
  auto i = a_dist(rng);

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

node_t tree_t::lookup(infoset_t infoset) const {
  assert (false);
}

void tree_t::create_node(infoset_t infoset) {
  assert (false);
}

auto tree_t::sample_sigma(infoset_t infoset) const -> sample_ret_t {
  assert (false);
}

}
