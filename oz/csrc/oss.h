#ifndef OZ_OSS_H
#define OZ_OSS_H

#include <vector>
#include <functional>

#include "game.h"

namespace oz {

class action_t {
};

class infoset_t {
 public:
  std::vector<action_t> actions() const;
  void foreach_action(std::function<void(action_t)> f);
};

bool operator==(const infoset_t &a, const infoset_t &b);
bool operator==(const action_t &a, const action_t &b);

struct action_prob_t {
  action_t a;
  prob_t pr_a;  // probability action was taken under policy
  prob_t rho1;  // probability of sampling action when targeted
  prob_t rho2;  // probability of sampling action
};

class history_t {
 public:
  void act(action_t a);
  history_t operator >>(action_t a);
  infoset_t infoset() const;
  player_t player() const;
  bool is_terminal() const;
  value_t utility(player_t player) const;
  action_prob_t sample_chance() const;
};

class sigma_t {
 public:
  action_prob_t sample_pr(infoset_t infoset);
  prob_t pr(infoset_t infoset, action_t a) const;
};

class node_t {
 public:
  sigma_t sigma_regret_matching();
  void accumulate_regret(action_t a, value_t r);
  void accumulate_average_strategy(action_t a, prob_t s);
};

class tree_t {
 public:
  node_t lookup(infoset_t infoset);
  std::tuple<action_prob_t, bool> sample_sigma(infoset_t infoset);
};

class oss_t {
 public:
  struct prefix_prob_t {
    prob_t pi_i;  // reach probability for search player
    prob_t pi_o;  // reach probability for opponent player and chance
    prob_t s1;    // probability of this sample when targeted
    prob_t s2;    // probability of this sample
  };

  struct suffix_prob_t {
    prob_t x;     // suffix probability
    prob_t l;     // tip-to-tail sample probability
    value_t u;    // value at the terminal
  };

  struct path_item_t {
    player_t player;
    infoset_t infoset;
    action_prob_t action_prob;
    prefix_prob_t prefix_prob;
  };

  class search_t {
   private:
    void step(action_prob_t ap);
    void walk(tree_t tree);
    void unwind(tree_t tree, suffix_prob_t prob);

    enum {IN_TREE, ROLLOUT_EVAL, PRIOR_EVAL, TERMINAL, END} state_;

    player_t search_player_;
    prefix_prob_t prefix_prob_;
    history_t history_;
    std::vector<path_item_t> path_;

    prob_t delta_;
  };

 private:
  tree_t tree_;
};

} // namespace oz

#endif // OZ_OSS_H
