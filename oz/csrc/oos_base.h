#ifndef OZ_OOS_BASE_H
#define OZ_OOS_BASE_H

#include "game.h"

#include <boost/container/small_vector.hpp>
#include <boost/container/flat_map.hpp>

namespace oz {

  struct action_prob_t {
    action_t a;
    prob_t pr_a;  // probability action was taken under policy
    prob_t rho1;  // probability of sampling action when targeted
    prob_t rho2;  // probability of sampling action not targeted
  };

  static constexpr int N_ACTIONS_SMALL = 16;
  using action_vector = boost::container::small_vector<action_t, N_ACTIONS_SMALL>;
  using prob_vector = boost::container::small_vector<prob_t, N_ACTIONS_SMALL>;

  using boost::container::flat_map;

  struct sample_ret_t {
    action_prob_t ap;
    bool out_of_tree = false;
  };

  using action_value_map_t = flat_map<action_t, value_t>;
  using action_prob_map_t  = flat_map<action_t, value_t>;

  using regret_map_t = action_value_map_t;

} // namespace oz

#endif // OZ_OOS_BASE_H
