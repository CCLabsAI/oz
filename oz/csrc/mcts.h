#ifndef OZ_MCTS_H
#define OZ_MCTS_H

#include "history.h"
#include "sigma.h"

#include <vector>
#include <unordered_map>

namespace oz { namespace mcts {

#include <boost/container/flat_map.hpp>

struct q_val_t {
  value_t w;
  int n;
};

struct node_t {
  flat_map<action_t, q_val_t> q;
  int n;
};

struct tree_t {
  sigma_t sigma_average() const;

  std::unordered_map<infoset_t, node_t> nodes;
};

class sigma_visits_t : public sigma_t::concept_t {
 public:
  explicit sigma_visits_t(const tree_t &tree): tree_(tree) { };
  prob_t pr(infoset_t infoset, action_t a) const override;
 private:
  const tree_t &tree_;
};

class search_t final {
 public:
  search_t(history_t history, prob_t c):
      history_(move(history)), c_(c) { };

  void select(const tree_t &tree, rng_t &rng); // walk from tip to leaf and updating path
  void create(tree_t &tree, rng_t &rng);       // add node to tree with zero values
  void playout_step(action_t a);               // step playout forward one ply
  void backprop(tree_t &tree);                 // unwind updates along path

  enum class state_t {
    SELECT,   // initial state
    CREATE,   // create node (with prior information)
    PLAYOUT,  // waiting for playout policy evaluation
    BACKPROP, // history is terminal, waiting to apply updates
    FINISHED
  };

  state_t state() const { return state_; }
  const history_t &history() const { return history_; }

 private:
  void tree_step(action_t a); // take one step in-tree and extend path
  void tree_step(action_t a, node_t *node); // take one step in-tree and extend path

  struct path_item_t {
    action_t a;
    player_t player;
    node_t *node;
  };

  using path_t = std::vector<path_item_t>;

  state_t state_ = state_t::SELECT;
  history_t history_;
  path_t path_;

  prob_t c_;
//  player_t search_player_;

}; // class search_t

void search(history_t h, int n_iter, tree_t &tree, rng_t &rng,
            const prob_t c);

}} // namespace oz::mcts

#endif // OZ_MCTS_H
