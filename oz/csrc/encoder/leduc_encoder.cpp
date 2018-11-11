#include <cassert>

#include "util.h"
#include "leduc_encoder.h"

namespace oz {

using namespace std;
using namespace at;

static auto cast_infoset(const infoset_t &infoset)
  -> const leduc_poker_t::infoset_t &
{
  // TODO this whole thing is still somewhat distressing
  return infoset.cast<leduc_poker_t::infoset_t>();
}

void leduc_encoder_t::card_one_hot(card_t card, ta_t &x_a, int i) {
  switch (card) {
    case card_t::Jack:
      x_a[i+0] = 1.0;
      break;
    case card_t::Queen:
      x_a[i+1] = 1.0;
      break;
    case card_t::King:
      x_a[i+2] = 1.0;
      break;
    case card_t::NA:
      break;
    default: assert (false);
  }
}

void leduc_encoder_t::action_one_hot(action_t action, ta_t &x_a, int i) {
  switch (action) {
    case action_t::Raise:
      x_a[i+0] = 1.0;
      break;
    case action_t::Call:
      x_a[i+1] = 1.0;
      break;
    default: assert (false);
  }
}

void leduc_encoder_t::rounds_one_hot(const leduc_poker_t::action_vector_t &actions,
                                     ta_t &x_a, int i)
{
  int round_n = 0, action_n = 0;
  for (const auto &a : actions) {
    switch (a) {
      case action_t::Raise:
      case action_t::Call:
        action_one_hot(a, x_a, i + round_n*ROUND_SIZE + action_n*ACTION_SIZE);
        action_n++;
        break;
      case action_t::NextRound:
        round_n++;
        action_n = 0;
        break;
      case action_t::Fold:
      default:
        assert (false);
    }

    if (round_n > N_ROUNDS) {
      break;
    }
  }
}

void leduc_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
  // TODO write tests
  Expects(x.size(0) == encoding_size());

  const auto &game_infoset = cast_infoset(infoset);
  Expects(game_infoset.player != CHANCE);

  x.zero_();
  auto x_a = x.accessor<nn_real_t, 1>();

  int pos = 0;

  card_one_hot(game_infoset.hand, x_a, pos);
  pos += CARD_SIZE;

  card_one_hot(game_infoset.board, x_a, pos);
  pos += CARD_SIZE;

  rounds_one_hot(game_infoset.history, x_a, pos);
}

static int action_to_idx(leduc_poker_t::action_t action) {
  using action_t = leduc_encoder_t::action_t;

  switch (action) {
    case action_t::Raise:
      return 0;
      break;
    case action_t::Call:
      return 1;
      break;
    case action_t::Fold:
      return 2;
      break;
    default:
      Ensures(false);
      return 0;
  }
}

void leduc_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {
  const auto actions = infoset.actions();
  auto x_a = x.accessor<nn_real_t, 1>();

  // NB don't zero the action probabilites, but leave them unchanged
  // allowing the caller to, place NaN there for illegal actions
  // x.zero_();
  for (const auto &action : actions) {
    const auto a_leduc = action.cast<leduc_poker_t::action_t>();
    int i = action_to_idx(a_leduc);
    x_a[i] = sigma.pr(infoset, action);
  }
}


auto leduc_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
{
  const auto actions = infoset.actions();
  auto m = map<oz::action_t, prob_t>();
  auto x_a = x.accessor<nn_real_t, 1>();

  for (const auto &action : actions) {
    const auto a_leduc = action.cast<leduc_poker_t::action_t>();
    int i = action_to_idx(a_leduc);
    m[action] = x_a[i];
  }

  return m;
}

auto leduc_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
{
  Expects(cast_infoset(infoset).player != CHANCE);
  Expects(x.size(0) == max_actions());

  const auto actions = infoset.actions();
  auto weights = vector<prob_t>(actions.size());

  auto x_a = x.accessor<nn_real_t, 1>();

  transform(begin(actions), end(actions), begin(weights),
            [&](const oz::action_t &action) -> prob_t {
    const auto a_leduc = action.cast<leduc_poker_t::action_t>();
    switch (a_leduc) {
      case action_t::Raise:
        return x_a[0];
      case action_t::Call:
        return x_a[1];
      case action_t::Fold:
        return x_a[2];
      default:
        assert (false);
        return 0;
    }
  });

  Expects(all_greater_equal_zero(weights));
  auto total = accumulate(begin(weights), end(weights), (prob_t) 0);

  auto a_dist = discrete_distribution<>(begin(weights), end(weights));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = (total > 0) ? weights[i]/total : (prob_t) 1.0 / actions.size();
  auto rho1 = pr_a, rho2 = pr_a;

  Ensures(pr_a >= 0 && pr_a <= 1);
  return { a, pr_a, rho1, rho2 };
}

} // namespace oz
