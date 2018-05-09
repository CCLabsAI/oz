#include "mcts.h"

#include <utility>

#include <boost/container/pmr/global_resource.hpp>

namespace oz { namespace mcts {

using boost::container::pmr::new_delete_resource;

struct sample_ret_t {
  bool out_of_tree;
  action_t a;
  node_t *node;
};

static
sample_ret_t sample_tree(const tree_t &tree,
                         const infoset_t &infoset,
                         params_t params,
                         rng_t &rng);

static
action_t sample_node(const node_t &node,
                     params_t params,
                     rng_t &rng);

static
action_t sample_chance(const history_t &history,
                       rng_t &rng);

static
action_t sample_action(const history_t &history,
                       rng_t &rng);

void search_t::tree_step(action_t a) {
  Expects(history_.player() == CHANCE);
  tree_step(a, nullptr);
}

void search_t::tree_step(action_t a, node_t *node) {
  const auto active_player = history_.player();

  path_.emplace_back(path_item_t {
    a, active_player, node
  });

  history_.act(a);
}

void search_t::select(const tree_t &tree, rng_t &rng) {
  Expects(state_ == state_t::SELECT);

  while (state_ == state_t::SELECT) {
    if (history_.is_terminal()) {
      state_ = state_t::BACKPROP;
    }
    else if (history_.player() == CHANCE) {
      const auto a = sample_chance(history_, rng);
      tree_step(a);
    }
    else {
      const auto infoset = history_.infoset();
      const auto r = sample_tree(tree, infoset, params_, rng);

      if (r.out_of_tree) {
        state_ = state_t::CREATE;
      }
      else {
        tree_step(r.a, r.node);
      }
    }
  }

  Ensures(state_ == state_t::CREATE || state_ == state_t::BACKPROP);
}

void search_t::create(tree_t &tree, rng_t &rng) {
  Expects(state_ == state_t::CREATE);
  Expects(history_.player() != CHANCE);
  Expects(!history_.is_terminal());

  const auto infoset = history_.infoset();
  const auto actions = infoset.actions();

  auto p = tree.nodes.emplace(infoset, node_t { });
  node_t &node = p.first->second;

  node.q.reserve(actions.size());
  for (const action_t a : infoset.actions()) {
    node.q[a]; // create empty entry
  }

  tree_step(sample_node(node, params_, rng), &node);

  if (history_.is_terminal()) {
    state_ = state_t::BACKPROP;
  }
  else {
    state_ = state_t::PLAYOUT;
  }

  Ensures(state_ == state_t::PLAYOUT || state_ == state_t::BACKPROP);
}

void search_t::playout_step(action_t a) {
  Expects(state_ == state_t::PLAYOUT);

  history_.act(a);

  if (history_.is_terminal()) {
    state_ = state_t::BACKPROP;
  }

  Ensures(state_ == state_t::PLAYOUT || state_ == state_t::BACKPROP);
}

void search_t::backprop(tree_t &tree) {
  Expects(state_ == state_t::BACKPROP);
  Expects(history_.is_terminal());

  const value_t u = history_.utility(P1);

  for (auto i = rbegin(path_); i != rend(path_); ++i) {
    const path_item_t &path_item = *i;
    const action_t a = path_item.a;
    const player_t player = path_item.player;
    node_t &node = *path_item.node;

    if (player == CHANCE) {
      Expects(path_item.node == nullptr);
      continue;
    }

    const value_t u_rel = relative_utility(player, u);

    auto &q = node.q[a];
    q.w += u_rel;
    q.n += 1;

    node.n += 1;
  }

  state_ = state_t::FINISHED;
}

value_t q_val_t::v_uct(int N, prob_t c) const {
  if (n == 0) {
    return std::numeric_limits<value_t>::infinity();
  }
  else {
    value_t v_avg = ((value_t) w / n);
    value_t v_exp = c * sqrt(log((value_t) N) / (value_t) n);
    return v_avg + v_exp;
  }
}

static
sample_ret_t sample_tree(const tree_t &tree,
                         const infoset_t &infoset,
                         const params_t params,
                         rng_t &rng)
{
  const auto it = tree.nodes.find(infoset);

  if (it != end(tree.nodes)) {
    auto &node = it->second;
    const auto a = sample_node(node, params, rng);

    // save a node pointer, used to modify the tree during backprop
    auto *node_ptr = const_cast<node_t*>(&it->second);
    return { false, a, node_ptr };
  }
  else {
    return { true };
  }
}

static
action_t sample_node(const node_t &node,
                     const params_t params,
                     rng_t &rng)
{
  const auto uct_value =
      [&](const decltype(node.q)::value_type &p) -> value_t {
        const q_val_t &q = p.second;
        return q.v_uct(node.n, params.c);
      };

  auto it = max_element_by(begin(node.q), end(node.q), uct_value);
  Expects(it != end(node.q));

  const auto a_best = it->first;
  return a_best;
}

static
action_t sample_chance(const history_t &history,
                       rng_t &rng)
{
  const auto ap = oz::sample_chance(history, rng, new_delete_resource());
  return ap.a;
}

static
action_t sample_action(const history_t &history,
                       rng_t &rng)
{
  const auto ap = oz::sample_action(history, rng);
  return ap.a;
}

sigma_t tree_t::sigma_average() const {
  return make_sigma<sigma_visits_t>(*this);
}

prob_t sigma_visits_t::pr(infoset_t infoset, action_t a) const {
  const auto it = tree_.nodes.find(infoset);
  if (it != end(tree_.nodes)) {
    const node_t &node = tree_.nodes.at(infoset);
    return (prob_t) node.q.at(a).n / node.n;
  }
  else {
    const auto n = infoset.actions().size();
    return (prob_t) 1.0 / n;
  }
}

static
void search_iter(history_t h,
                 tree_t &tree,
                 params_t params,
                 rng_t &rng)
{
  using state_t = search_t::state_t;

  search_t s(move(h), params);

  while (s.state() != state_t::FINISHED) {
    switch (s.state()) {
      case state_t::SELECT:
        s.select(tree, rng);
        break;
      case state_t::CREATE:
        s.create(tree, rng);
        break;
      case state_t::PLAYOUT:
        s.playout_step(oz::mcts::sample_action(s.history(), rng));
        break;
      case state_t::BACKPROP:
        s.backprop(tree);
        break;
      case state_t::FINISHED:
        break;
    }
  }
}

void search(history_t h, int n_iter, tree_t &tree,
            params_t params, rng_t &rng)
{
  Expects(n_iter >= 0);

  for(int i = 0; i < n_iter; i++) {
    search_iter(h, tree, params, rng);
  }
}


}} // namespace oz::mcts
