#include <cassert>

#include "util.h"
#include "holdem_encoder.h"

namespace oz {

using namespace std;
using namespace at;

static auto cast_infoset(const infoset_t &infoset)
  -> const holdem_poker_t::infoset_t &
{
  // TODO this whole thing is still somewhat distressing
  return infoset.cast<holdem_poker_t::infoset_t>();
}

void holdem_encoder_t::card_one_hot(card_t card, ta_t &x_a, int i) {
  if (card != holdem_poker_t::CARD_NA) {
    Expects(holdem_poker_t::CARD_MIN <= card && card <= holdem_poker_t::CARD_MAX);
    auto j = card - holdem_poker_t::CARD_MIN;
    auto d = std::div(j, holdem_poker_t::N_RANKS);

    int rank = d.rem;
    int suit = d.quot;

    x_a[i+rank] = 1.0;
    x_a[i+holdem_poker_t::N_RANKS+suit] = 1.0;
  }
}

void holdem_encoder_t::action_one_hot(action_t action, ta_t &x_a, int i) {
  switch (action) {
    case action_t::Raise:
      x_a[i+0] = 1.0;
      break;
    case action_t::Call:
      x_a[i+1] = 1.0;
      break;
    default:
      assert(false);
  }
}

inline static int player_idx(const player_t p) {
  Expects(p == P1 || p == P2);
  switch (p) {
    case P1: return 0;
    case P2: return 1;
    default: return 0; // NB not reachable
  }
}

void holdem_encoder_t::rounds_one_hot(player_t player, const holdem_poker_t::action_vector_t &actions,
                                      ta_t &x_a, int i)
{
  static constexpr int PLAYER_STRIDE = N_ROUNDS*MAX_ROUND_ACTIONS*ACTION_SIZE;
  static constexpr int ROUNDE_STRIDE = MAX_ROUND_ACTIONS*ACTION_SIZE;
  static constexpr int ACTION_STRIDE = ACTION_SIZE;

  int p = player_idx(holdem_poker_t::FIRST_PLAYER[0]), player_relative;
  int round_n = 0, action_n = 0;
  for (const auto &a : actions) {
    switch (a) {
      case action_t::Raise:
      case action_t::Call:
        player_relative = (p + player_idx(player)) % 2;
        action_one_hot(a, x_a, i + player_relative*PLAYER_STRIDE + round_n*ROUNDE_STRIDE + action_n*ACTION_STRIDE);
        action_n++;
        p = (p + 1) % 2;
        break;
      case action_t::NextRound:
        round_n++;
        action_n = 0;
        p = player_idx(holdem_poker_t::FIRST_PLAYER[round_n]);
        break;
      case action_t::Fold:
      default:
        assert(false);
    }

    if (round_n > N_ROUNDS) {
      break;
    }
  }
}

void holdem_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
  Expects(x.size(0) == encoding_size());

  const auto &game_infoset = cast_infoset(infoset);
  Expects(game_infoset.player != CHANCE);

  x.zero_();
  auto x_a = x.accessor<nn_real_t, 1>();

  int pos = 0;

  card_one_hot(game_infoset.hand[0], x_a, pos);
  pos += CARD_SIZE;
  card_one_hot(game_infoset.hand[1], x_a, pos);
  pos += CARD_SIZE;

  for (auto c : game_infoset.board) {
    card_one_hot(c, x_a, pos);
    pos += CARD_SIZE;
  }

  pos = MAX_CARDS*CARD_SIZE; // start of round encoding
  rounds_one_hot(game_infoset.player, game_infoset.history, x_a, pos);
}

// TODO remove duplication

static int action_to_idx(holdem_poker_t::action_t action) {
  using action_t = holdem_encoder_t::action_t;

  switch (action) {
    case action_t::Raise:
      return 0;
    case action_t::Call:
      return 1;
    case action_t::Fold:
      return 2;
    default:
      assert(false);
      return 0;
  }
}

void holdem_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {
  const auto actions = infoset.actions();
  auto x_a = x.accessor<nn_real_t, 1>();

  // NB don't zero the action probabilities, but leave them unchanged
  // allowing the caller to, place NaN there for illegal actions
  // x.zero_();
  for (const auto &action : actions) {
    const auto a_holdem = action.cast<holdem_poker_t::action_t>();
    int i = action_to_idx(a_holdem);
    x_a[i] = static_cast<nn_real_t>(sigma.pr(infoset, action));
  }
}


auto holdem_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
{
  const auto actions = infoset.actions();
  auto m = map<oz::action_t, prob_t>();
  auto x_a = x.accessor<nn_real_t, 1>();

  for (const auto &action : actions) {
    const auto a_holdem = action.cast<holdem_poker_t::action_t>();
    int i = action_to_idx(a_holdem);
    m[action] = x_a[i];
  }

  return m;
}

auto holdem_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
{
  Expects(cast_infoset(infoset).player != CHANCE);
  Expects(x.size(0) == max_actions());

  const auto actions = infoset.actions();
  auto weights = vector<prob_t>(actions.size());

  auto x_a = x.accessor<nn_real_t, 1>();

  transform(begin(actions), end(actions), begin(weights),
            [&](const oz::action_t &action) -> prob_t {
    const auto a_holdem = action.cast<holdem_poker_t::action_t>();
    switch (a_holdem) {
      case action_t::Raise:
        return x_a[0];
      case action_t::Call:
        return x_a[1];
      case action_t::Fold:
        return x_a[2];
      default:
        assert(false);
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
