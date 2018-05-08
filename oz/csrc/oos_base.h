#ifndef OZ_OOS_BASE_H
#define OZ_OOS_BASE_H

#include "game.h"

#include <boost/container/small_vector.hpp>

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

} // namespace oz

#endif // OZ_OOS_BASE_H
