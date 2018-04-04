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

void oss_t::search_t::tree_step(action_prob_t ap) {
  assert (state_ == state_t::SELECT || state_ == state_t::CREATE);
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

void oss_t::search_t::select(const tree_t& tree, rng_t &rng) {
  assert (state_ == state_t::SELECT);

  while (state_ == state_t::SELECT) {
    const auto acting_player = history_.player();

    if (history_.is_terminal()) {
      enter_backprop();
    }
    else if (acting_player == CHANCE) {
      const auto ap = history_.sample_chance(rng);
      tree_step(ap);
    }
    else {
      const auto infoset = history_.infoset();
      const auto r = tree.sample_sigma(infoset, rng);

      if (r.out_of_tree) {
        state_ = state_t::CREATE;
      }
      else {
        tree_step(r.ap);
      }
    }
  }
}

void oss_t::search_t::enter_backprop() {
  oz::prob_t s1 = prefix_prob_.s1;
  oz::prob_t s2 = prefix_prob_.s2;

  suffix_prob_.x = 1.0;
  suffix_prob_.l = delta_ * s1 + (1.0 - delta_) * s2;
  suffix_prob_.u = history_.utility(search_player_);

  state_ = oz::oss_t::search_t::state_t::BACKPROP;
}

void oss_t::search_t::create(tree_t& tree, rng_t &rng) {
  assert (state_ == state_t::CREATE);
  assert (history_.player() != CHANCE);
  assert (!history_.is_terminal());

  const auto infoset = history_.infoset();
  tree.create_node(infoset);
  
  const auto r = tree.sample_sigma(infoset, rng);
  assert (!r.out_of_tree);

  tree_step(r.ap);

  if (history_.is_terminal()) {
    enter_backprop();
  }
  else {
    state_ = state_t::PLAYOUT;    
  }
}

void oss_t::search_t::playout_step(action_prob_t ap) {
  assert (state_ == state_t::PLAYOUT);
  
  history_.act(ap.a);
  suffix_prob_.x *= ap.pr_a;

  // TODO this is sort of a weird special case, make more elegant
  if (history_.is_terminal()) {
    prob_t s1 = prefix_prob_.s1 * suffix_prob_.x;
    prob_t s2 = prefix_prob_.s2 * suffix_prob_.x;

    suffix_prob_.l = delta_ * s1 + (1.0 - delta_) * s2;
    suffix_prob_.u = history_.utility(search_player_);
    
    state_ = state_t::BACKPROP;
  }
}

void oss_t::search_t::backprop(tree_t& tree) {
  assert (state_ == state_t::BACKPROP);
  assert (history_.is_terminal());

  prob_t c;
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

    if (active_player == CHANCE) { // TODO make this more elegant
      continue;
    }

    auto &node = tree.lookup(infoset);

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
        const prob_t s = (pi_o * sigma.pr(infoset, a_prime)) / q;
        node.accumulate_average_strategy(a_prime, s);
      }
    }
  }

  state_ = state_t::FINISHED;
}

auto sigma_t::concept_t::sample_pr(infoset_t infoset, rng_t& rng) const ->
action_prob_t {
  auto actions = infoset.actions();
  auto probs = vector<prob_t>(actions.size());
  
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

action_prob_t sigma_t::sample_eps(infoset_t infoset,
                                  prob_t eps,
                                  rng_t &rng) const {
  auto d = uniform_real_distribution<>();
  prob_t u = d(rng);

  auto actions = infoset.actions();
  auto p = (prob_t) 1/actions.size();

  if (u <= eps) {
    auto dd = uniform_int_distribution<>(0, actions.size()-1);
    auto i = dd(rng);
    auto a = actions[i];

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

node_t::node_t(std::vector<action_t> actions) {
  auto regret_in = inserter(regrets_, regrets_.end());
  transform(begin(actions), end(actions), regret_in,
            [](const action_t &a) { return make_pair(a, 0); });

  auto avg_in = inserter(average_stratergy_, average_stratergy_.end());
  transform(begin(actions), end(actions), avg_in,
            [](const action_t &a) { return make_pair(a, 0); });
}

auto history_t::sample_chance(rng_t& rng) -> action_prob_t {
  // FIXME use real probs
  const auto actions = infoset().actions();

  auto probs = vector<prob_t>(actions.size());

  fill(begin(probs), end(probs), (prob_t) 1/actions.size());

  auto a_dist = discrete_distribution<>(begin(probs), end(probs));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = probs[i];
  auto rho1 = pr_a;
  auto rho2 = pr_a;

  return { a, pr_a, rho1, rho2 };
}

void tree_t::create_node(infoset_t infoset) {
  nodes_.emplace(infoset, node_t(infoset.actions()));
}

auto tree_t::sample_sigma(infoset_t infoset, rng_t &rng) const -> sample_ret_t {
  const auto it = nodes_.find(infoset);

  if (it == end(nodes_)) {
    return { { }, true };
  }
  else {
    const auto &node = lookup(infoset);
    const auto sigma = node.sigma_regret_matching();

    const auto ap = sigma.sample_eps(infoset, 0.5, rng);
    return { ap, false };
  }
}

auto tree_t::sigma_average() const -> sigma_t {
  return make_sigma<sigma_average_t>(*this);
}

template <typename T>
T rectify(T x) {
  return x > 0 ? x : 0;
}

auto sigma_regret_t::pr(infoset_t infoset, action_t a) const -> prob_t {
  auto sum_positive = accumulate(
    begin(regrets_), end(regrets_), (value_t) 0,
    [](const auto &r, const auto &x) {
        return r + rectify(x.second);
    });

  prob_t p;
  if (sum_positive > 0) {
    auto r = regrets_.at(a);
    p = r > 0 ? r/sum_positive : 0;
  }
  else {
    p = (prob_t) 1/infoset.actions().size();
  }

  assert(p >= 0 && p <= 1);
  return p;
}

auto sigma_regret_t::sample_pr(infoset_t infoset, rng_t &rng) const
  -> action_prob_t {
  auto actions = vector<action_t>(regrets_.size());
  auto weights = vector<prob_t>(regrets_.size());

  transform(begin(regrets_), end(regrets_), begin(actions),
            [](const auto &x) { return x.first; });

  transform(begin(regrets_), end(regrets_), begin(weights),
            [](const auto &x) { return rectify(x.second); });

  auto total = accumulate(begin(weights), end(weights), (value_t) 0);

  assert (actions.size() > 0);
  assert (weights.size() > 0);

  int i;
  if (total > 0) {
    auto a_dist = discrete_distribution<>(begin(weights), end(weights));
    i = a_dist(rng);
  }
  else {
    auto a_dist = uniform_int_distribution<>(0, weights.size()-1);
    i = a_dist(rng);
  }

  auto a = actions[i];
  auto pr_a = total > 0 ? weights[i]/total : (prob_t) 1/weights.size();
  auto rho1 = pr_a;
  auto rho2 = pr_a;

  assert (pr_a >= 0 && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

auto sigma_average_t::pr(infoset_t infoset, action_t a) const -> prob_t {
  const auto actions = infoset.actions();
  const auto &node = tree_.lookup(infoset);
  const auto total = accumulate(begin(actions), end(actions), (prob_t) 0,
    [&](const auto &r, const auto &a_prime){ return r + node.average_strategy(a_prime); });

  auto p = node.average_strategy(a) / total;
  assert (p >= 0 && p <= 1);
  return p;
};

void oss_t::search_step(history_t h,
                        player_t player,
                        tree_t &tree,
                        rng_t &rng) {
  search_t s(move(h), player);
  while (s.state() != search_t::state_t::FINISHED) {
    switch (s.state()) {
      case search_t::state_t::SELECT:
        s.select(tree, rng);
        break;
      case search_t::state_t::CREATE:
        s.create(tree, rng);
        break;
      case search_t::state_t::PLAYOUT:
        // FIXME handle chance nodes properly
        {
          auto infoset = s.infoset();
          auto actions = infoset.actions();
          auto d = uniform_int_distribution<>(1, actions.size() - 1);
          auto i = d(rng);
          auto a = actions[i];
          auto pr_a = (prob_t) 1 / actions.size();
          s.playout_step(action_prob_t{ a, pr_a, pr_a, pr_a });
        }
        break;
      case search_t::state_t::BACKPROP:
        s.backprop(tree);
        break;
      case search_t::state_t::FINISHED:
        break;
    }
  }
}

void oss_t::search(history_t h,
                   int n_iter,
                   tree_t &tree,
                   rng_t &rng) {
  for(int i = 0; i < n_iter; i++) {
    search_step(h, P1, tree, rng);
    search_step(h, P2, tree, rng);
  }
}

}
