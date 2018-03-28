#ifndef OZ_BEST_RESPONSE_H
#define OZ_BEST_RESPONSE_H

#include <unordered_map>

#include "oss.h"

namespace oz {

using br_stats_t = std::unordered_map<std::pair<std::string, action_t>, value_t>;

value_t exploitability(history_t h, sigma_t sigma);

value_t gebr(history_t h, player_t i,
             sigma_t sigma);

value_t gebr(history_t h, player_t i,
             sigma_t sigma, std::vector<int> depths);

value_t gebr_pass2(history_t h, player_t i,
                   int d, int l, prob_t pi_o,
                   sigma_t sigma, br_stats_t& t, br_stats_t& b);

std::vector<int> infoset_depths(history_t h);

} // namespace oz

#endif // OZ_BEST_RESPONSE_H
