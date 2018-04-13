#include <cassert>

#include "util.h"
#include "encoder.h"

namespace oz {

using namespace std;
using namespace at;

auto leduk_encoder_t::cast_infoset(const infoset_t &infoset)
  -> const leduk_poker_t::infoset_t &
{
  // TODO this whole thing is still somewhat distressing
  return infoset.as<leduk_poker_t::infoset_t>();
}

void leduk_encoder_t::card_one_hot(card_t card, ta_t &x_a, int i) {
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

void leduk_encoder_t::action_one_hot(action_t action, ta_t &x_a, int i) {
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

void leduk_encoder_t::rounds_one_hot(const vector<action_t> &actions,
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


// TODO write tests
void leduk_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
  const auto &game_infoset = cast_infoset(infoset);
  assert (game_infoset.player != CHANCE);

  auto x_a = x.accessor<nn_real_t, 1>();

  for (int i = 0; i < x_a.size(0); i++) {
    x_a[i] = 0;
  }

  int pos = 0;

  card_one_hot(game_infoset.hand, x_a, pos);
  pos += CARD_SIZE;

  card_one_hot(game_infoset.board, x_a, pos);
  pos += CARD_SIZE;

  rounds_one_hot(game_infoset.history, x_a, pos);
}

auto leduk_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
{
  const auto actions = infoset.actions();
  auto m = map<oz::action_t, prob_t>();
  auto x_a = x.accessor<nn_real_t, 1>();

  for (const auto &action : actions) {
    prob_t p;
    const auto a_leduk = action.template as<leduk_poker_t::action_t>();
    switch (a_leduk) {
      case action_t::Raise:
        p = x_a[0];
        break;
      case action_t::Call:
        p = x_a[1];
        break;
      case action_t::Fold:
        p = x_a[2];
        break;
      default:
        assert (false);
        p = 0;
    }

    m[action] = p;
  }

  return m;
}

auto leduk_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
{
  assert (cast_infoset(infoset).player != CHANCE);
  assert (x.size(0) == max_actions());

  const auto actions = infoset.actions();
  auto weights = vector<prob_t>(actions.size());

  auto x_a = x.accessor<nn_real_t, 1>();

  transform(begin(actions), end(actions), begin(weights),
            [&](const oz::action_t &action) -> prob_t {
    switch (action.template as<leduk_poker_t::action_t>()) {
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

  assert (all_greater_than_zero(weights));
  auto total = accumulate(begin(weights), end(weights), (prob_t) 0);

  auto a_dist = discrete_distribution<>(begin(weights), end(weights));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = weights[i]/total;
  auto rho1 = pr_a, rho2 = pr_a;

  assert (total >= 0);
  assert (pr_a >= 0 && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

} // namespace oz
