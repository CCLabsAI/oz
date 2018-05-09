#ifndef OZ_BEST_RESPONSE_H
#define OZ_BEST_RESPONSE_H

#include <unordered_map>

#include "history.h"
#include "sigma.h"

namespace oz {

using std::pair;
using std::vector;
using std::unordered_map;

struct q_val_t {
  value_t t = 0;
  value_t b = 0;
  value_t v() { return b > 0 ? t / b : 0; }
};

using q_info_t = pair<infoset_t, action_t>;
using q_stats_t = unordered_map<q_info_t, q_val_t>;

value_t exploitability(history_t h, sigma_t sigma);

value_t gebr(history_t h, player_t i,
             sigma_t sigma);

value_t gebr_pass2(history_t h, player_t i,
                   int d, int l, prob_t pi_o,
                   sigma_t sigma, q_stats_t& tb);

vector<int> infoset_depths(history_t h, player_t i);

} // namespace oz

#endif // OZ_BEST_RESPONSE_H
