#ifndef OZ_BEST_RESPONSE_H
#define OZ_BEST_RESPONSE_H

#include <unordered_map>

#include "oss.h"

namespace oz {

struct q_val_t {
  value_t t = 0;
  value_t b = 0;
  value_t v() { return b > 0 ? t / b : 0; }
};

using q_info_t = std::pair<infoset_t, action_t>;
using q_stats_t = std::unordered_map<q_info_t, q_val_t>;

value_t exploitability(history_t h, sigma_t sigma);

value_t gebr(history_t h, player_t i,
             sigma_t sigma);

value_t gebr(history_t h, player_t i,
             sigma_t sigma, std::vector<int> depths);

value_t gebr_pass2(history_t h, player_t i,
                   int d, int l, prob_t pi_o,
                   sigma_t sigma, q_stats_t& tb);

std::vector<int> infoset_depths(history_t h, player_t i);

} // namespace oz

#endif // OZ_BEST_RESPONSE_H
