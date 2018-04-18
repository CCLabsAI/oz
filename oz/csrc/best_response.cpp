#include <cassert>
#include <algorithm>
#include <iterator>
#include <vector>
#include <set>

#include "util.h"
#include "hash.h"

#include "best_response.h"

namespace oz {

using namespace std;

auto exploitability(history_t h, sigma_t sigma) -> value_t {
  value_t v1 = gebr(h, P1, sigma);
  value_t v2 = gebr(h, P2, sigma);

  return v1 + v2;
}

static auto actions(const history_t& h) -> vector<action_t> {
  if (h.player() == CHANCE) {
    const auto actions_pr = h.chance_actions();
    return keys(actions_pr);
  }
  else {
    const auto infoset = h.infoset();
    return infoset.actions();
  }
}

auto gebr(history_t h, player_t i, sigma_t sigma) -> value_t {
  q_stats_t tb;

  for (int d : infoset_depths(h, i)) {
    gebr_pass2(h, i, d, 0, 1.0, sigma, tb);
  }

  // final pass should maximize at every depth, so: d = -1
  value_t v = gebr_pass2(h, i, -1, 0, 1.0, sigma, tb);
  return v;
}

auto gebr_pass2(history_t h, player_t i,
                int d, int l, prob_t pi_o,
                sigma_t sigma, q_stats_t& tb) -> value_t {

  if (h.is_terminal()) {
    return h.utility(i);
  }

  const auto player = h.player();

  if (player == CHANCE) {
    const auto actions_pr = h.chance_actions();
    Expects(!actions_pr.empty());

    value_t v_chance = 0;
    for (const auto& ap : actions_pr) {
      const action_t a = ap.first;
      const prob_t pr_a = ap.second;

      value_t v_a = gebr_pass2(h >> a, i,
                               d, l + 1, pi_o * pr_a,
                               sigma, tb);
      v_chance += pr_a * v_a;
    }

    return v_chance;
  }

  Expects(player != CHANCE);
  const auto infoset = h.infoset();
  const auto actions = infoset.actions();

  if (player == i && l > d) {
    Expects(!actions.empty());

    auto value_lookup = [&](const action_t& a) -> value_t {
      return tb.at({ infoset, a }).v();
    };

    auto a_best = max_element_by(begin(actions), end(actions), value_lookup);
    return gebr_pass2(h >> *a_best, i,
                      d, l + 1, pi_o,
                      sigma, tb);
  }

  value_t v = 0;
  for (const auto& a : actions) {
    prob_t pi_prime_o = pi_o;
    if (player != i) {
      pi_prime_o *= sigma.pr(infoset, a);
    }

    value_t v_prime = gebr_pass2(h >> a, i,
                                 d, l + 1, pi_prime_o,
                                 sigma, tb);

    if (player != i) {
      v += sigma.pr(infoset, a) * v_prime;
    }
    else if (player == i && l == d) {
      auto& q = tb[{ infoset, a }];
      q.t += v_prime * pi_o;
      q.b += pi_o;
    }
  }

  return v;
}

void walk_infosets(const history_t& h, player_t i, set<int>& depths, int l) {
  if (h.is_terminal()) {
    return;
  }

  const auto player = h.player();

  if (player == i) {
    depths.insert(l);
  }

  for (const auto& a : actions(h)) {
    walk_infosets(h >> a, i, depths, l + 1);
  }
}

auto infoset_depths(history_t h, player_t i) -> vector<int> {
  set<int> depths;
  walk_infosets(h, i, depths, 0);
  vector<int> depth_list(begin(depths), end(depths));
  sort(begin(depth_list), end(depth_list), greater<>());
  return depth_list;
}

} // namespace oz
