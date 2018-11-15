#ifndef OZ_TIC_TAC_TOES_H
#define OZ_TIC_TAC_TOES_H


#include "game.h"

#include <cassert>
#include <array>
#include <set>
#include <vector>
#include <bitset>

#include <boost/container/static_vector.hpp>

namespace oz {

  using std::move;
  using std::string;
  using std::array;
  using std::set;
  using std::vector;
  using std::vector;
  using std::bitset;
  
  using boost::container::static_vector;

  class tic_tac_toes_t final : public game_t {
    public:

      static constexpr int MAX_SQUARES = 9;
    
      enum class action_t {

      /* actions for P1 and P2 :
       P1 is the one playing X, P2 plays 0
       The actions are fill_1 to fill_9 numbered as follow:
       
       3 6 9
       2 5 8
       1 4 7
       
      */
      
      fill_1 = 0,
      fill_2,
      fill_3,
      fill_4,
      fill_5,
      fill_6,
      fill_7,
      fill_8,
      fill_9 = 8,
          
      NextRound = 100

      // There is no chance player
    };

    
    using action_vector_t = static_vector<action_t, MAX_SQUARES>;
   

    
    struct infoset_t : public oz::infoset_t::concept_t {
      const player_t player;
      const action_vector_t pieces_P1;
      const action_vector_t pieces_P2;
      const action_vector_t history;
      const int action_number;
      
      const int is_terminal_flag;
      

      infoset_t(player_t player,action_vector_t pieces_P1, action_vector_t pieces_P2, action_vector_t history, int action_number, int is_terminal_flag):
          player(player),pieces_P1(move(pieces_P1)), pieces_P2(move(pieces_P2)), history(move(history)), action_number(action_number),is_terminal_flag(is_terminal_flag){ }

      actions_list_t actions() const override;
      string str() const override;
      bool is_equal(const concept_t &that) const override;
      size_t hash() const override;
    };

    void act_(action_t a);

    void act(oz::action_t a) override { act_(a.cast<action_t>()); }
    oz::infoset_t infoset() const override;
    player_t player() const override { return player_; }
       bool is_terminal() const override;
    value_t utility(player_t player) const override;
    oz::infoset_t infoset(oz::infoset_t::allocator_t alloc) const override;
    action_prob_map_t chance_actions() const override;
    std::string str() const override;
    

    std::unique_ptr<game_t> clone() const override {
      return std::make_unique<tic_tac_toes_t>(*this);
    }


    action_prob_map_t chance_actions(action_prob_allocator_t alloc) const override;
    
    static constexpr int MAX_VALUE_ACTION = 9;
   

  private:
    
    // First player is P1
    player_t player_ = P1;
    
    int round_ = 0;
    int action_number = 0;
    int is_terminal_flag = 0;
    action_vector_t pieces_P1_;
    action_vector_t pieces_P2_;
    int turn_ = 0;
    action_vector_t history_;
    
    
    
    player_t other_player() const {
      assert(player_ == P1 || player_ == P2);
      return player_ == P1 ? P2 : P1;
    }

    inline static int player_idx(player_t p) {
      assert(p == P1 || p == P2);
      switch (p) {
        case P1: return 0;
        case P2: return 1;
        default: return -1; // should not be reachable
      }
    }
    static int is_winning_move_vector(action_vector_t moves);

    
    
  public:
    const action_vector_t &pieces_P1() const { return pieces_P1_; }
    const action_vector_t &pieces_P2() const { return pieces_P2_; }
    
    int turn() const { return turn_; }
    
    const action_vector_t &history() const { return history_; }
    
  };

} // namespace oz

#endif // OZ_TIC_TAC_TOES_H
