#include <cassert>
#include <iterator>
#include <set>

#include "util.h"
#include "best_response.h"

namespace oz {

using namespace std;

auto exploitability(history_t h, sigma_t sigma) -> value_t {
  auto depths = infoset_depths(h);
  value_t v1 = gebr(h, P1, sigma, depths);
  value_t v2 = gebr(h, P2, sigma, depths);

  return v1 + v2;
}

auto gebr(history_t h, player_t i, sigma_t sigma) -> value_t {
  const auto depths = infoset_depths(h);
  return gebr(move(h), i, move(sigma), depths);
}

auto gebr(history_t h, player_t i,
          sigma_t sigma, vector<int> depths) -> value_t {
  q_stats_t tb;

  for (const auto& d : depths) {
    gebr_pass2(h, i, d, 0, 1.0, sigma, tb);
  }

  // final pass should maximize at every depth, so: d = -1
  value_t v = gebr_pass2(move(h), i, -1, 0, 1.0, move(sigma), tb);
  return v;
}

auto gebr_pass2(history_t h, player_t i,
                int d, int l, prob_t pi_o,
                sigma_t sigma, q_stats_t& tb) -> value_t {
  if (h.is_terminal()) {
    return h.utility(i);
  }

  const auto player = h.player();
  const auto infoset = h.infoset();
  const auto actions = infoset.actions();

  assert(!actions.empty());

  if (player == CHANCE) {
    value_t v_chance = 0;
    for (const auto& a : actions) {
      auto pr_a = (prob_t) 1 / actions.size(); // FIXME
      value_t v_a = gebr_pass2(h >> a, i,
                               d, l + 1, pi_o * pr_a,
                               sigma, tb);
      v_chance += pr_a * v_a;
    }

    return v_chance;
  }

  if (player == i && l > d) {
    auto value_fn = [&](const action_t& a) -> value_t {
      return tb.at({ infoset, a }).v();
    };

    auto a_best = max_element_by(begin(actions), end(actions), value_fn);
    return gebr_pass2(h >> *a_best, i,
                      d, l + 1, pi_o,
                      sigma, tb);
  }

  value_t v = 0;
  for (const auto& a : actions) {
    prob_t pi_prime_o = pi_o;
    if (player != i) {
      pi_prime_o = pi_o * sigma.pr(infoset, a);
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

auto walk_infosets(const history_t& h, set<int>& depths, int l) -> void {
  if (h.is_terminal()) {
    return;
  }

  const auto player = h.player();
  const auto infoset = h.infoset();

  if (player != CHANCE) {
    depths.insert(l);
  }

  for (const auto& a : infoset.actions()) {
    walk_infosets(h >> a, depths, l + 1);
  }
}

auto infoset_depths(history_t h) -> vector<int> {
  set<int> depths;
  walk_infosets(h, depths, 0);
  vector<int> depth_list(begin(depths), end(depths));
  sort(begin(depth_list), end(depth_list), greater<>());
  return depth_list;
}

}
