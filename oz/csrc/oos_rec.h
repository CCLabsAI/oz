#ifndef OZ_OOS_REC_H
#define OZ_OOS_REC_H

/////////////////////////
// UNDER CONSTRUCTION  //
/////////////////////////

#include "oos.h"

namespace oz { namespace rec {

struct action_prob_t {
  action_t a;
  prob_t pr_a;
};

struct action_prob2_t {
  action_t a;
  prob_t s1;
  prob_t s2;
};

#if 0

class sigma_t {
 public:
  prob_t pr(infoset_t infoset, action_t a);
  action_prob_t sample(infoset_t infoset);
};

class tree_t {
 public:
  class node_t {
   public:
    sigma_t sigma_regret_matching();
    void update_regret(action_t a, value_t r);
    void update_average_strategy(action_t a, value_t s);
  };

  struct lookup_ret_t {
    node_t node;
    bool out_of_tree;
  };

  lookup_ret_t lookup(infoset_t infoset);
};

#endif

template <class tree_t, class sigma_t>
class oos_t {
  struct walk_ret_t {
    prob_t x;
    prob_t l;
    value_t u;
  };

 public:
  walk_ret_t walk(history_t h,
                  prob_t pi_i, prob_t pi_o,
                  prob_t s1, prob_t s2, player_t i);

  action_prob2_t sample(history_t h, sigma_t sigma);
  action_prob2_t sample_chance(history_t h);
  action_prob_t sample_action(history_t h, sigma_t sigma);
  walk_ret_t playout(history_t h, prob_t s, sigma_t sigma);

 private:
  real_t delta_;
  real_t eps_;
  sigma_t sigma_playout_;
  tree_t tree_;
};

} } // namespace oz::rec

#endif // OZ_OOS_REC_H
