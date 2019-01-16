//
// Created by Michela on 22/5/18.
//

#include <cassert>

#include "util.h"
#include "liars_dice_encoder.h"

namespace oz {

  using namespace std;
  using namespace torch;

  static auto cast_infoset(const infoset_t &infoset)
  -> const liars_dice_t::infoset_t &
  {
    // TODO this whole thing is still somewhat distressing
    return infoset.cast<liars_dice_t::infoset_t>();
  }

  void liars_dice_encoder_t::face_one_hot(face_t face, ta_t &x_a, int i) {
    switch (face) {
      case face_t ::face_1:
        x_a[i+0] = 1.0;
        break;
      case face_t::face_2:
        x_a[i+1] = 1.0;
        break;
      case face_t::face_3:
        x_a[i+2] = 1.0;
        break;
      case face_t::face_4:
        x_a[i+3] = 1.0;
        break;
      case face_t::face_5:
        x_a[i+4] = 1.0;
        break;
      case face_t::face_star:
        x_a[i+5] = 1.0;
        break;
      case face_t::NA:
        break;
      default: assert (false);
    }
  }

  void liars_dice_encoder_t::action_one_hot(action_t action, ta_t &x_a, int i) {
    switch (action) {
      case action_t::Raise_0face:
        x_a[i+0] = 1.0;
        break;
      case action_t::Raise_1face:
        x_a[i+1] = 1.0;
        break;
      case action_t::Raise_2face:
        x_a[i+2] = 1.0;
        break;
      case action_t::Raise_3face:
        x_a[i+3] = 1.0;
        break;
      case action_t::Raise_4face:
        x_a[i+4] = 1.0;
        break;
      case action_t::Raise_5face:
        x_a[i+5] = 1.0;
        break;
      case action_t::Raise_0dice:
        x_a[i+6] = 1.0;
        break;
      case action_t::Raise_1dice:
        x_a[i+7] = 1.0;
        break;
      case action_t::Raise_2dice:
        x_a[i+8] = 1.0;
        break;
      case action_t::Raise_3dice:
        x_a[i+9] = 1.0;
        break;
      case action_t::Raise_4dice:
        x_a[i+10] = 1.0;
        break;
      default: assert (false);
    }
  }

  void liars_dice_encoder_t::rounds_one_hot(const liars_dice_t::action_vector_t &actions,
                                   ta_t &x_a, int i)
  {
    int round_n = 0, action_n = 0;
    for (const auto &a : actions) {
      switch (a) {
        case action_t::Raise_0face:
        case action_t::Raise_1face:
        case action_t::Raise_2face:
        case action_t::Raise_3face:
        case action_t::Raise_4face:
        case action_t::Raise_5face:
        case action_t::Raise_0dice:
        case action_t::Raise_1dice:
        case action_t::Raise_2dice:
        case action_t::Raise_3dice:
        case action_t::Raise_4dice:


          action_one_hot(a, x_a, i + round_n * liars_dice_encoder_t::ROUND_SIZE + action_n * liars_dice_encoder_t::ACTION_SIZE);
          action_n++;
          break;
        case action_t::NextRound:
          round_n++;
          action_n = 0;
          break;
        case action_t::Call_liar:
        default:
          assert (false);
      }

      if (round_n > N_ROUNDS) {
        break;
      }
    }
  }

  void liars_dice_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
    Expects(x.size(0) == encoding_size());

    const auto &game_infoset = cast_infoset(infoset);
    Expects(game_infoset.player != CHANCE);

    x.zero_();
    auto x_a = x.accessor<nn_real_t, 1>();

    int pos = 0;

    face_one_hot(game_infoset.face1, x_a, pos);
    pos += DICE_SIZE;

    if (liars_dice_t::N_DICES == 2) {
      face_one_hot(game_infoset.face2, x_a, pos);
      pos += DICE_SIZE;
    }

    rounds_one_hot(game_infoset.history, x_a, pos);
  }

  static int action_to_idx(liars_dice_encoder_t::action_t action) {
    using action_t = liars_dice_encoder_t::action_t;

    switch (action) {
      case action_t::Raise_0face:
        return 0;
        break;
      case action_t::Raise_1face:
        return 1;
        break;
      case action_t::Raise_2face:
        return 2;
        break;
      case action_t::Raise_3face:
        return 3;
        break;
      case action_t::Raise_4face:
        return 4;
        break;
      case action_t::Raise_5face:
        return 5;
        break;
      case action_t::Raise_0dice:
        return 6;
        break;
      case action_t::Raise_1dice:
        return 7;
        break;
      case action_t::Raise_2dice:
        return 8;
        break;
      case action_t::Raise_3dice:
        return 9;
        break;
      case action_t::Raise_4dice:
        return 10;
        break;
      case action_t::Call_liar:
        return 11;
        break;
      default:
        Ensures(false);
        return 0;
    }
  }

  void liars_dice_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {
    const auto actions = infoset.actions();
    auto x_a = x.accessor<nn_real_t, 1>();

    x.zero_();

    for (const auto &action : actions) {
      const auto a_liar = action.cast<liars_dice_encoder_t::action_t>();
      int i = action_to_idx(a_liar);
      x_a[i] = sigma.pr(infoset, action);
    }
  }


  auto liars_dice_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
  {
    const auto actions = infoset.actions();
    auto m = map<oz::action_t, prob_t>();
    auto x_a = x.accessor<nn_real_t, 1>();

    for (const auto &action : actions) {
      const auto a_liar_dice = action.cast<liars_dice_encoder_t::action_t>();
      int i = action_to_idx(a_liar_dice);
      m[action] = x_a[i];
    }

    return m;
  }

  auto liars_dice_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
  {
    Expects(cast_infoset(infoset).player != CHANCE);
    Expects(x.size(0) == max_actions());

    const auto actions = infoset.actions();
    auto weights = vector<prob_t>(actions.size());

    auto x_a = x.accessor<nn_real_t, 1>();

    transform(begin(actions), end(actions), begin(weights),
              [&](const oz::action_t &action) -> prob_t {
                const auto a_liar_dice = action.cast<liars_dice_encoder_t::action_t>();
                switch (a_liar_dice) {
                  case action_t::Raise_0face:
                    return x_a[0];
                  case action_t::Raise_1face:
                    return x_a[1];
                  case action_t::Raise_2face:
                    return x_a[2];
                  case action_t::Raise_3face:
                    return x_a[3];
                  case action_t::Raise_4face:
                    return x_a[4];
                  case action_t::Raise_5face:
                    return x_a[5];
                  case action_t::Raise_0dice:
                    return x_a[6];
                  case action_t::Raise_1dice:
                    return x_a[7];
                  case action_t::Raise_2dice:
                    return x_a[8];
                  case action_t::Raise_3dice:
                    return x_a[9];
                  case action_t::Raise_4dice:
                    return x_a[10];
                  case action_t::Call_liar:
                    return x_a[11];
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
    auto pr_a = weights[i]/total;
    auto rho1 = pr_a, rho2 = pr_a;

    Ensures(total >= 0);
    Ensures(pr_a >= 0 && pr_a <= 1);

    return { a, pr_a, rho1, rho2 };
  }

} // namespace oz
