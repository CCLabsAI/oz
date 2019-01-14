//
// Created by Michela on 22/5/18.
//

#include <cassert>

#include "util.h"
#include "tic_tac_toe_encoder.h"

namespace oz {

  using namespace std;
  using namespace at;

  static auto cast_infoset(const infoset_t &infoset) -> const tic_tac_toe_t::infoset_t & {
    return infoset.cast<tic_tac_toe_t::infoset_t>();
  }

  
    
  int action_to_idx(const tic_tac_toe_t::action_t a){
      switch(a){
          case tic_tac_toe_t::action_t::fill_1 :
              return 0;
          case tic_tac_toe_t::action_t::fill_2 :
              return 1;
          case tic_tac_toe_t::action_t::fill_3 :
              return 2;
          case tic_tac_toe_t::action_t::fill_4 :
              return 3;
          case tic_tac_toe_t::action_t::fill_5 :
              return 4;
          case tic_tac_toe_t::action_t::fill_6 :
              return 5;
          case tic_tac_toe_t::action_t::fill_7 :
              return 6;
          case tic_tac_toe_t::action_t::fill_8 :
              return 7;
          case tic_tac_toe_t::action_t::fill_9 :
              return 8;
          default: assert(false);
              return -1;
          
          }
    }
    
  void tic_tac_toe_encoder_t::action_one_hot(action_t action, ta_t &x_a, int i) {
    x_a[action_to_idx(action)] = 1.0;
    
  }
   

  void tic_tac_toe_encoder_t::rounds_one_hot(const tic_tac_toe_t::action_vector_t &actions, ta_t &x_a, int i){
        
      int round_n = 0, action_n = 0;
      tic_tac_toe_t::action_vector_t round_actions;
        
      for (const auto &a : actions) {
            switch (a) {
                case action_t::fill_1:
                case action_t::fill_2:
                case action_t::fill_3:
                case action_t::fill_4:
                case action_t::fill_5:
                case action_t::fill_6:
                case action_t::fill_7:
                case action_t::fill_8:
                case action_t::fill_9:
                    round_actions.push_back(a);
                    break;
                case action_t::NextRound:
                    action_n += (5 - round_actions.size());
                    for (unsigned int index = 0; index < round_actions.size(); ++index){
                        action_one_hot(round_actions[index], x_a, i + round_n*ROUND_SIZE + action_n*ACTION_SIZE);
                        action_n++;
                    }
                    round_n++;
                    action_n = 0;
                    round_actions.clear();
                    break;
                default:
                    assert (false);
            }

            assert (round_n <= N_ROUNDS);
        }
  }
    
    
  void tic_tac_toe_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
    
    Expects(x.size(0) == encoding_size());
    
    
    const auto &game_infoset = cast_infoset(infoset);
    
    bool player_1 = game_infoset.player == P1;
    auto x_a = x.accessor<nn_real_t, 1>();

    int pos = 0;
      
    // Player 1
    if (player_1){
        rounds_one_hot(game_infoset.pieces_P1, x_a, pos);
      
    }
    // Player 2
    else {
          rounds_one_hot(game_infoset.pieces_P2, x_a, pos);
    }
   
  }


  void tic_tac_toe_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {

    
    const auto actions = infoset.actions();
    
    auto x_a = x.accessor<nn_real_t, 1>();
      
    for (const auto &action : actions) {
      const auto a_tic_tac_toe = action.cast<tic_tac_toe_encoder_t::action_t>();
      int i = action_to_idx(a_tic_tac_toe);
      x_a[i] = sigma.pr(infoset, action);
    }
    

  }


  auto tic_tac_toe_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
  {

    
    const auto actions = infoset.actions();
    auto m = map<oz::action_t, prob_t>();
    auto x_a = x.accessor<nn_real_t, 1>();

    for (const auto &action : actions) {
      const auto a_tic_tac_toe = action.cast<tic_tac_toe_encoder_t::action_t>();
      int i = action_to_idx(a_tic_tac_toe);
      m[action] = x_a[i];
    }

    
    return m;
  }

  auto tic_tac_toe_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
  {


    Expects(cast_infoset(infoset).player != CHANCE);
    Expects(x.size(0) == max_actions());

    const auto actions = infoset.actions();
    auto weights = vector<prob_t>(actions.size());

    auto x_a = x.accessor<nn_real_t, 1>();

    transform(begin(actions), end(actions), begin(weights),
              [&](const oz::action_t &action) -> prob_t {
                const auto a_tic_tac_toe = action.cast<tic_tac_toe_encoder_t::action_t>();
                switch (a_tic_tac_toe) {
                  case action_t::fill_1:
                    return x_a[0];
                  case action_t::fill_2:
                    return x_a[1];
                  case action_t::fill_3:
                    return x_a[2];
                  case action_t::fill_4:
                    return x_a[3];
                  case action_t::fill_5:
                    return x_a[4];
                  case action_t::fill_6:
                    return x_a[5];
                  case action_t::fill_7:
                    return x_a[6];
                  case action_t::fill_8:
                    return x_a[7];
                  case action_t::fill_9:
                    return x_a[8];
                  
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
