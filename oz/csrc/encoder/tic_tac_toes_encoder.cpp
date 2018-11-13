//
// Created by Michela on 22/5/18.
//

#include <cassert>

#include "util.h"
#include "tic_tac_toes_encoder.h"

namespace oz {

  using namespace std;
  using namespace at;

  static auto cast_infoset(const infoset_t &infoset)
  -> const tic_tac_toes_t::infoset_t &
  {
    // TODO this whole thing is still somewhat distressing
    return infoset.cast<tic_tac_toes_t::infoset_t>();
  }

  
  void tic_tac_toes_encoder_t::action_one_hot(int action, ta_t &x_a, int i) {
    x_a[i + action] = 1.0;
    
   
  }

  void tic_tac_toes_encoder_t::rounds_one_hot(int action_number,
                                              array<int, tic_tac_toes_t::MAX_SQUARES> tot_moves_P1,
                                              array<int, tic_tac_toes_t::MAX_SQUARES> tot_moves_P2,
                                              ta_t &x_a, int i)
  {

    int round_n = 0, action_n = 0;
    // player 2
    if (action_number % 2 == 0){
      
      for (int index = 0; index < tic_tac_toes_t::MAX_SQUARES; index ++){
        if (tot_moves_P2[index] - 11 == round_n){
          action_one_hot(index, x_a, i + round_n * tic_tac_toes_encoder_t::ROUND_SIZE + action_n * tic_tac_toes_encoder_t::ACTION_SIZE);
          action_n++;
        }
      }
      action_n = 4;
      for (int index = 0; index < tic_tac_toes_t::MAX_SQUARES; index ++){
        if (tot_moves_P2[index] == 1){
          action_one_hot(index, x_a, i + round_n * tic_tac_toes_encoder_t::ROUND_SIZE + action_n * tic_tac_toes_encoder_t::ACTION_SIZE);
          round_n ++;
          
          
        }
      }
    }
    // player 1
    else {
      
      for (int index = 0; index < tic_tac_toes_t::MAX_SQUARES; index ++){
        if (tot_moves_P1[index] - 11 == round_n){
          
          action_one_hot(index, x_a, i + round_n * tic_tac_toes_encoder_t::ROUND_SIZE + action_n * tic_tac_toes_encoder_t::ACTION_SIZE);
          action_n++;
        }
      }
        
      action_n = 4;
      for (int index = 0; index < tic_tac_toes_t::MAX_SQUARES; index ++){
        if (tot_moves_P1[index] == 1){
          
          
          action_one_hot(index, x_a, 35 + i + round_n * tic_tac_toes_encoder_t::ROUND_SIZE + action_n * tic_tac_toes_encoder_t::ACTION_SIZE);
          round_n ++;
        }
        
      }
    }
    
  }

  void tic_tac_toes_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
    
    Expects(x.size(0) == encoding_size());

    const auto &game_infoset = cast_infoset(infoset);
    const auto &tot_moves_P1 = game_infoset.tot_moves_P1;
    const auto &tot_moves_P2 = game_infoset.tot_moves_P2;
    const auto &action_number = game_infoset.action_number;
    
    Expects(game_infoset.player != CHANCE);

    x.zero_();
    auto x_a = x.accessor<nn_real_t, 1>();

    int pos = 0;
    if (action_number > 4){
      for(unsigned int i=0; i<9; i++){
        cout << tot_moves_P1[i] << " ";
        cout << tot_moves_P2[i] << endl;
        
      }
      cout << endl;
      
      getchar();
    }

    rounds_one_hot(action_number, tot_moves_P1, tot_moves_P2, x_a, pos);
    

  }

  static int action_to_idx(tic_tac_toes_encoder_t::action_t action) {

    
    using action_t = tic_tac_toes_encoder_t::action_t;
    unsigned int legal_action_base = 36;
    
    switch (action) {
      case action_t::fill_1:
        return legal_action_base;
      case action_t::fill_2:
        return legal_action_base + 1;
      case action_t::fill_3:
        return legal_action_base + 2;
      case action_t::fill_4:
        return legal_action_base + 3;
      case action_t::fill_5:
        return legal_action_base + 4;
      case action_t::fill_6:
        return legal_action_base + 5;
      case action_t::fill_7:
        return legal_action_base + 6;
      case action_t::fill_8:
        return legal_action_base + 7;
      case action_t::fill_9:
        return legal_action_base + 8;
      default:
        Ensures(false);
        return 0;
    }
    
  }

  void tic_tac_toes_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {

    
    const auto actions = infoset.actions();
    const auto &game_infoset = cast_infoset(infoset);
    const auto &tot_moves_P1 = game_infoset.tot_moves_P1;
    const auto &tot_moves_P2 = game_infoset.tot_moves_P2;
    const auto &action_number = game_infoset.action_number;
    
    auto x_a = x.accessor<nn_real_t, 1>();
    unsigned int legal_action_base = 36;


    /*for(unsigned int i = 0; i < MAX_SQUARES; i++){
      if (action_number % 2 == 0){
        for
      }
    }*/
    for (const auto &action : actions) {
      const auto a_tic_tac_toes = action.cast<tic_tac_toes_encoder_t::action_t>();
      int i = action_to_idx(a_tic_tac_toes);
      x_a[i] = sigma.pr(infoset, action);
    }
    

  }


  auto tic_tac_toes_encoder_t::decode(oz::infoset_t infoset, Tensor x)
  -> map<oz::action_t, real_t>
  {

    
    const auto actions = infoset.actions();
    auto m = map<oz::action_t, prob_t>();
    auto x_a = x.accessor<nn_real_t, 1>();

    for (const auto &action : actions) {
      const auto a_tic_tac_toes = action.cast<tic_tac_toes_encoder_t::action_t>();
      int i = action_to_idx(a_tic_tac_toes);
      m[action] = x_a[i];
    }

    
    return m;
  }

  auto tic_tac_toes_encoder_t::decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng)
  -> action_prob_t
  {


    Expects(cast_infoset(infoset).player != CHANCE);
    Expects(x.size(0) == max_actions());

    const auto actions = infoset.actions();
    auto weights = vector<prob_t>(actions.size());

    auto x_a = x.accessor<nn_real_t, 1>();

    transform(begin(actions), end(actions), begin(weights),
              [&](const oz::action_t &action) -> prob_t {
                const auto a_tic_tac_toes = action.cast<tic_tac_toes_encoder_t::action_t>();
                switch (a_tic_tac_toes) {
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
