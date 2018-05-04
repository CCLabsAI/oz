#include <cassert>

#include "util.h"
#include "goofspiel2_encoder.h"

namespace oz {

using namespace std;
using namespace at;

static auto cast_infoset(const infoset_t &infoset)
  -> const goofspiel2_t::infoset_t &
{
  return infoset.cast<goofspiel2_t::infoset_t>();
}

int goofspiel2_encoder_t::encoding_size() {
  // one hot encoding for each card and each bid
  // every card will eventually be bid at least once
  int bids_size = n_cards_*n_cards_;
  // one hot encoding for bid outcome (win, loss, draw)
  int wins_size = 3*n_cards_;

  return bids_size + wins_size;
}

int goofspiel2_encoder_t::max_actions() {
  return n_cards_; // could possibly bid 1 of N cards
}

void goofspiel2_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
  // TODO write tests
  Expects(x.size(0) == encoding_size());

  const auto &game_infoset = cast_infoset(infoset);
  const auto &bids = game_infoset.bids();
  const auto &wins = game_infoset.wins();
  const auto n_bids = static_cast<int>(bids.size());
  const auto n_wins = static_cast<int>(wins.size());
  const auto wins_pos = n_cards_ * n_cards_;


  Expects(n_bids == n_wins);

  x.zero_();
  auto x_a = x.accessor<nn_real_t, 1>();

  for (int n = 0; n < n_bids; n++) {
    const int card_idx = bids[n];
    x_a[n*n_cards_ + card_idx] = 1.0;
  }

  int n, pos;
  for (n = 0, pos = wins_pos; n < n_wins; n++, pos += 3) {
    switch (wins[n]) {
      case CHANCE:
        x_a[pos + 0] = 1.0;
        break;
      case P1:
        x_a[pos + 1] = 1.0;
        break;
      case P2:
        x_a[pos + 2] = 1.0;
        break;
    }
  }

  Ensures(pos <= encoding_size());
}

void goofspiel2_encoder_t::encode_sigma(infoset_t infoset,
                                        sigma_t sigma, Tensor x)
{
  const auto actions = infoset.actions();
  auto x_a = x.accessor<nn_real_t, 1>();

  x.zero_();
  for (const auto &action : actions) {
    const int a_goof = action.cast<goofspiel2_t::action_t>();
    x_a[a_goof] = sigma.pr(infoset, action);
  }
}


auto goofspiel2_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, prob_t>
{
  // TODO remove duplication
  const auto actions = infoset.actions();
  auto m = map<oz::action_t, prob_t>();
  auto x_a = x.accessor<nn_real_t, 1>();

  for (const auto &action : actions) {
    const int a_goof = action.cast<goofspiel2_t::action_t>();
    prob_t p = x_a[a_goof];
    m[action] = p;
  }

  return m;
}

auto goofspiel2_encoder_t::decode_and_sample(oz::infoset_t infoset,
                                             Tensor x, rng_t &rng)
  -> action_prob_t
{
  // TODO remove duplication
  Expects(x.size(0) == max_actions());

  const auto actions = infoset.actions();
  auto weights = vector<prob_t>(actions.size());

  auto x_a = x.accessor<nn_real_t, 1>();

  transform(begin(actions), end(actions), begin(weights),
            [&](const oz::action_t &action) -> prob_t {
    const int a_goof = action.cast<goofspiel2_t::action_t>();
    prob_t p = x_a[a_goof];
    return p;
  });

  Expects(all_greater_equal_zero(weights));
  auto total = accumulate(begin(weights), end(weights), (prob_t) 0);

  auto a_dist = discrete_distribution<>(begin(weights), end(weights));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = weights[i]/total;
  auto rho1 = pr_a, rho2 = pr_a;

  // TODO what if all the weights are 0?
  Ensures(total > 0);
  Ensures(pr_a >= 0 && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

} // namespace oz
