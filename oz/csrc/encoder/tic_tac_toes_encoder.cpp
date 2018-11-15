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

  /*void tic_tac_toes_encoder_t::rounds_one_hot(int action_number,
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
    
  }*/

  void tic_tac_toes_encoder_t::encode(oz::infoset_t infoset, Tensor x) {
    
    Expects(x.size(0) == encoding_size());
    
    
    const auto &game_infoset = cast_infoset(infoset);
    const auto &tot_moves_P1 = game_infoset.tot_moves_P1;
    const auto &tot_moves_P2 = game_infoset.tot_moves_P2;
    const auto &pieces_P1 = game_infoset.pieces_P1;
    const auto &pieces_P2 = game_infoset.pieces_P2;
    const auto &action_number = game_infoset.action_number;
    const auto category_pos = MAX_ACTIONS * MAX_ACTIONS;
    

    bool player_1 = game_infoset.player == P1;
    auto x_a = x.accessor<nn_real_t, 1>();

    int pos = 0;
    unsigned int found_flag = 0;
    
    // Encode the action of the player
    // Player 1
    if (player_1){
      const auto n_pieces = static_cast<int>(pieces_P1.size());
      for (int n = 0; n < n_pieces; n++) {
        const int piece_idx = pieces_P1[n];
        if (piece_idx < 10){
          x_a[n*MAX_ACTIONS + piece_idx - 1] = 1.0;
          
        }
        else {
          x_a[n*MAX_ACTIONS + piece_idx - 11] = 1.0;
        }
        
      }
      
      // Encoding the category of the action
      int n, pos;
      for (n = 0, pos = category_pos; n < MAX_ACTIONS; n++, pos += 2) {
        if (tot_moves_P1[n] == 1) {
            x_a[pos + 0] = 1.0;
        }
        
        else if (tot_moves_P1[n] == 2) {
            x_a[pos + 1] = 1.0;
        }
      }
    }
    // Player 2
    else {
      int n, pos;
      const auto n_pieces = static_cast<int>(pieces_P2.size());
      for (n = 0; n < n_pieces; n++) {
        const int piece_idx = pieces_P2[n];
        if (piece_idx < 10){
          x_a[n*MAX_ACTIONS + piece_idx - 1] = 1.0;
        }
        else {
          x_a[n*MAX_ACTIONS + piece_idx - 11] = 1.0;
        }
      }
      
      // Encoding the category of the action
      for (n = 0, pos = category_pos; n < MAX_ACTIONS; n++, pos += 2) {
        if (tot_moves_P2[n] == 1) {
          x_a[pos + 0] = 1.0;
        }
        
        else if (tot_moves_P2[n] == 2) {
          x_a[pos + 1] = 1.0;
        }
      }
      
    }
    
    

  }

  static int action_to_idx(tic_tac_toes_encoder_t::action_t action) {

    
    using action_t = tic_tac_toes_encoder_t::action_t;
    
    switch (action) {
      case action_t::fill_1:
        return 0;
      case action_t::fill_2:
        return 1;
      case action_t::fill_3:
        return 2;
      case action_t::fill_4:
        return 3;
      case action_t::fill_5:
        return 4;
      case action_t::fill_6:
        return 5;
      case action_t::fill_7:
        return 6;
      case action_t::fill_8:
        return 7;
      case action_t::fill_9:
        return 8;
      default:
        Ensures(false);
        return 0;
    }
    
  }

  void tic_tac_toes_encoder_t::encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) {

    
    const auto actions = infoset.actions();
    
    auto x_a = x.accessor<nn_real_t, 1>();
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
