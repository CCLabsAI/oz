#include <cassert>

#include "encoder.h"

namespace oz {

using namespace std;
using namespace at;

template <class T, class U>
auto assert_cast(U&& x) -> T {
#ifndef NDEBUG
  return dynamic_cast<T>(std::forward<U>(x));
#else
  return static_cast<T>(std::forward<U>(x));
#endif
};

using nn_real_t = float;
using ta_t = TensorAccessor<nn_real_t,1>;

inline void card_one_hot(leduk_poker_t::card_t card, ta_t &x_a, int i) {
  using card_t = leduk_poker_t::card_t;

  switch (card) {
    case card_t::Jack:
      x_a[i+0] = 1;
      break;
    case card_t::Queen:
      x_a[i+1] = 1;
      break;
    case card_t::King:
      x_a[i+2] = 1;
      break;
    case card_t::NA:
      break;
    default: assert (false);
  }
}

inline void action_one_hot(leduk_poker_t::action_t action, ta_t &x_a, int i) {
  using action_t = leduk_poker_t::action_t;

  switch (action) {
    case action_t::Raise:
      x_a[i+0] = 1;
      break;
    case action_t::Call:
      x_a[i+1] = 1;
      break;
    default: assert (false);
  }
}

// TODO write tests
void leduk_encoder_t::encode(infoset_t infoset, Tensor x) {
  const auto &game_infoset =
      assert_cast<const leduk_poker_t::infoset_t&>(infoset.get());
  assert (game_infoset.player != CHANCE);

  auto x_a = x.accessor<nn_real_t,1>();

  for (int i = 0; i < x_a.size(0); i++) {
    x_a[i] = 0;
  }

  int pos = 0;

  card_one_hot(game_infoset.hand, x_a, pos);
  pos += CARD_SIZE;

  card_one_hot(game_infoset.board, x_a, pos);
  pos += CARD_SIZE;

  int round_n = 0;
  int action_n = 0;
  for (const auto &a : game_infoset.history) {
    int ii = pos + round_n*ROUND_SIZE + action_n*ACTION_SIZE;
    switch (a) {
      case leduk_poker_t::action_t::Raise:
      case leduk_poker_t::action_t::Call:
        action_one_hot(a, x_a, ii);
        action_n++;
        break;
      case leduk_poker_t::action_t::NextRound:
        round_n++; action_n = 0;
        break;
      case leduk_poker_t::action_t::Fold:
      default:
        assert (false);
    }

    if (round_n > leduk_poker_t::N_ROUNDS) {
      break;
    }
  }
}

auto leduk_encoder_t::decode_and_sample(infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t {
  const auto &game_infoset =
      assert_cast<const leduk_poker_t::infoset_t&>(infoset.get());
  assert (game_infoset.player != CHANCE);
  assert (x.size(0) == 3);

  const auto actions = infoset.actions();
  auto weights = vector<prob_t>(actions.size());

  auto x_a = x.accessor<nn_real_t,1>();

  transform(begin(actions), end(actions), begin(weights),
            [&](const action_t &action) -> prob_t {
    switch (action.template as<leduk_poker_t::action_t>()) {
      case leduk_poker_t::action_t::Raise:
        return x_a[0];
      case leduk_poker_t::action_t::Call:
        return x_a[1];
      case leduk_poker_t::action_t::Fold:
        return x_a[2];
      default:
        assert (false);
        return 0;
    }
  });

  auto total = accumulate(begin(weights), end(weights), (value_t) 0);

  auto a_dist = discrete_distribution<>(begin(weights), end(weights));
  auto i = a_dist(rng);

  auto a = actions[i];
  auto pr_a = weights[i]/total;
  auto rho1 = pr_a, rho2 = pr_a;

  assert (total > 0);
  assert (pr_a >= 0 && pr_a <= 1);

  return { a, pr_a, rho1, rho2 };
}

} // namespace oz
