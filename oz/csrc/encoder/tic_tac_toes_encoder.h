//
// Created by Michela on 22/5/18.
//

#ifndef OZ_TIC_TAC_TOES_ENCODER_H
#define OZ_TIC_TAC_TOES_ENCODER_H



#include <ATen/ATen.h>

#include "encoder.h"

#include "game.h"
#include "oos.h"

#include "games/tic_tac_toes.h"

namespace oz {

  using std::vector;
  using std::map;
  using std::array;
  using at::Tensor;

  class tic_tac_toes_encoder_t final : public encoder_t {
  public:
    using action_t = tic_tac_toes_t::action_t;

    int encoding_size() override { return ENCODING_SIZE; };
    int max_actions() override { return MAX_ACTIONS; };
    void encode(oz::infoset_t infoset, Tensor x) override;
    void encode_sigma(infoset_t infoset, sigma_t sigma, Tensor x) override;
    map<oz::action_t, prob_t> decode(oz::infoset_t infoset, Tensor x) override;
    action_prob_t decode_and_sample(oz::infoset_t infoset, Tensor x, rng_t &rng) override;

  private:
    using nn_real_t = float;
    using ta_t = at::TensorAccessor<nn_real_t, 1>;


    static void action_one_hot(int action, ta_t &x_a, int i);
    static void rounds_one_hot(
                               int player_idx,
                               const tic_tac_toes_t::action_vector_t &actions, array<int, tic_tac_toes_t::MAX_SQUARES> tot_moves_P1, array<int, tic_tac_toes_t::MAX_SQUARES> tot_moves_P2, ta_t &x_a, int i);


    static constexpr int N_ROUNDS = 1;
    
    static constexpr int ACTION_SIZE = 9 ;
    static constexpr int ROUND_SIZE = 5 * ACTION_SIZE;

    static constexpr int ENCODING_SIZE =  ROUND_SIZE * 5;
    static constexpr int MAX_ACTIONS = 9;


  };

};


#endif //OZ_TIC_TAC_TOES_ENCODER_H
