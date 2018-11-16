#include "tic_tac_toes.h"
#include "hash.h"

#include <cassert>
#include <memory>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>

bool show_move = 0;

namespace oz {

  using namespace std;
    
    
   int actions_to_idx(const tic_tac_toes_t::action_t a){
      switch(a){
          case tic_tac_toes_t::action_t::fill_1 :
              return 0;
          case tic_tac_toes_t::action_t::fill_2 :
              return 1;
          case tic_tac_toes_t::action_t::fill_3 :
              return 2;
          case tic_tac_toes_t::action_t::fill_4 :
              return 3;
          case tic_tac_toes_t::action_t::fill_5 :
              return 4;
          case tic_tac_toes_t::action_t::fill_6 :
              return 5;
          case tic_tac_toes_t::action_t::fill_7 :
              return 6;
          case tic_tac_toes_t::action_t::fill_8 :
              return 7;
          case tic_tac_toes_t::action_t::fill_9 :
              return 8;
          default: return -1;
          
          }
              
  }
    
    

  void tic_tac_toes_t::act_(action_t a) {
    
    if (action_number > MAX_VALUE_ACTION ) {
      throw std::invalid_argument("maximum number of moves reached");
    }
    if (player_idx(player_) == 0){
      if (action_number == MAX_VALUE_ACTION){
        is_terminal_flag = is_winning_move_vector(pieces_P1_);
        if (is_terminal_flag == 0)
          is_terminal_flag = 2;
      }
    }
    else{
      if (action_number == MAX_VALUE_ACTION){
        is_terminal_flag = is_winning_move_vector(pieces_P2_);
        if (is_terminal_flag == 0)
          is_terminal_flag = 2;
      }
    }
    
    history_.push_back(a);
    //cout << infoset().str() << endl;
      
      
    // case current player is Player 1
    if (player_idx(player_) == 0){
        
        // calculate opponent past actions 
        int past_actions[MAX_SQUARES] = {0,0,0,0,0,0,0,0};
        for (const auto& a : pieces_P2_){
               int idx = actions_to_idx(a);
               past_actions[idx] = 1;
           }
        unsigned int current_action_idx = actions_to_idx(a);
        
        // case illegal move
        if(past_actions[current_action_idx] == 1)
            pieces_P1_.push_back(a);
        
        // Legal move
        else{
            pieces_P1_.push_back(a);
            pieces_P1_.push_back(action_t::NextRound);
            
            history_.push_back(action_t::NextRound);
            
            
            is_terminal_flag = is_winning_move_vector(pieces_P1_);
    
                                             
            player_ = other_player();
            action_number += 1;
            if (is_terminal_flag == 0) {
                if (action_number == MAX_VALUE_ACTION){
                    is_terminal_flag = 2;
                }
            }
            
        }
    }
      
    // case current player is Player 2
    else {
        // calculate opponent past actions 
        int past_actions[MAX_SQUARES] = {0,0,0,0,0,0,0,0};
        for (const auto& a : pieces_P1_){
               int idx = actions_to_idx(a);
               past_actions[idx] = 1;
           }
        unsigned int current_action_idx = actions_to_idx(a);
        
        // case illegal move
        if(past_actions[current_action_idx] == 1)
            pieces_P2_.push_back(a);
        
        // Legal move
        else{
               
            pieces_P2_.push_back(a);
            pieces_P2_.push_back(action_t::NextRound);
            history_.push_back(action_t::NextRound);
            is_terminal_flag = is_winning_move_vector(pieces_P2_);
            player_ = other_player();
            action_number += 1;
            if (is_terminal_flag == 0) {
                if (action_number == MAX_VALUE_ACTION){
                is_terminal_flag = 2;
                }
            }
            
        }
    }
  }
    



  
  auto tic_tac_toes_t::utility(player_t player) const -> value_t {
    assert (is_terminal());

    value_t u;
    if (is_terminal_flag == 2){
      u = 0;
    }
    else{
      if (action_number % 2 == 0)
        u = -1;
      else
        u = 1;
    }
      
    if (show_move == 1){
        cout << "Result " << u <<endl;
        getchar();
        }
      
    return u;
    
      
  }

  
      
  auto tic_tac_toes_t::is_winning_move_vector(action_vector_t moves) -> int{
    
    
    unsigned int end_of_the_game = 0;
    int tot_moves[MAX_VALUE_ACTION] = {0,0,0,0,0,0,0,0,0};
    
    unsigned int moves_size = moves.size();
    for (unsigned int i = 0; i < moves_size; ++i){
            if(actions_to_idx(moves[i]) == -1){
                tot_moves[actions_to_idx(moves[i - 1])] = 1;
            }
    }
      
    
        
        if (tot_moves[0] == 1){
          /* case
           x - -
           x - -
           x - -
           */
          if (tot_moves[1] == 1 and tot_moves[2] == 1){
            end_of_the_game = 1;
          }
          /* case
           - - -
           - - -
           x x x
           */
          else if (tot_moves[3] == 1 and tot_moves[6] == 1){
            end_of_the_game = 1;
          }
          /* case
           - - x
           - x -
           x - -
           */
          else if (tot_moves[4] == 1 and tot_moves[8] == 1){
            end_of_the_game = 1;
          }
        }
        /* case
         - - -
         x x x
         - - -
         */
        if (tot_moves[1] == 1 and tot_moves[4] == 1 and tot_moves[7] == 1){
          end_of_the_game = 1;
        }
        if (tot_moves[2] == 1) {
          
          /* case
           x x x
           - - -
           - - -
           */
          if (tot_moves[5] == 1 and tot_moves[8] == 1){
            end_of_the_game = 1;
          }
          /* case
           x - -
           - x -
           - - x
           */
          else if (tot_moves[4] == 1 and tot_moves[6] == 1){
            end_of_the_game = 1;
          }
          
        }
        /* case
         - x -
         - x -
         - x -
         */
        
        if (tot_moves[3] == 1 and tot_moves[4] == 1 and tot_moves[5] == 1){
          end_of_the_game = 1;
          
        }
        /* case
         - - x
         - - x
         - - x
         */
        if (tot_moves[6] == 1 and tot_moves[7] == 1 and tot_moves[8] == 1){
          end_of_the_game = 1;
          
        }
        return end_of_the_game;
        
      }
  auto tic_tac_toes_t::infoset() const -> oz::infoset_t {
    Expects(player() != CHANCE);
    return make_infoset<infoset_t>(player_, pieces_P1_, pieces_P2_, history_, action_number, is_terminal_flag);

  }
  

  auto tic_tac_toes_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t
  {
    Expects(player() != CHANCE);
    
    return allocate_infoset<infoset_t, oz::infoset_t::allocator_t>
        (alloc,
         player_, pieces_P1_, pieces_P2_, history_,
         action_number, is_terminal_flag);
  }

  auto tic_tac_toes_t::is_terminal() const -> bool {
    return is_terminal_flag;
    
  }


  

  auto tic_tac_toes_t::infoset_t::actions() const -> actions_list_t {
    
    actions_list_t actions;
    int past_actions[MAX_SQUARES] = {0,0,0,0,0,0,0,0};
     
    
    // Player P1
    if (player_idx(player) == 0){
      if (history.empty()){
          for (unsigned int i = 0; i < MAX_SQUARES; ++i)
              actions.push_back(make_action(i));
       }
       else {
           for (const auto& a : pieces_P1){
               int idx = actions_to_idx(a);
               past_actions[idx] = 1;
           }
           for (unsigned int i = 0; i < MAX_SQUARES; ++i){
               if(past_actions[i] == 0)
                   actions.push_back(make_action(i));
           }
           
           
       }
    }
    // Player P2
    else{
      if (history.empty()){
          for (unsigned int i = 0; i < MAX_SQUARES; ++i)
              actions.push_back(make_action(i));
       }
       else {
           for (const auto& a : pieces_P2){
               int idx = actions_to_idx(a);
               past_actions[idx] = 1;
           }
           for (unsigned int i = 0; i < MAX_SQUARES; ++i){
               if(past_actions[i] == 0)
                   actions.push_back(make_action(i));
           }
       }
      }
    
    return actions;
  }



  auto tic_tac_toes_t::chance_actions(action_prob_allocator_t alloc) const -> action_prob_map_t {
    Expects(player() == CHANCE);
    
    return { };

  }

  auto tic_tac_toes_t::chance_actions() const -> action_prob_map_t {
    return tic_tac_toes_t::chance_actions({ });
  }

  
  static std::ostream& operator << (std::ostream& os,
                                   const tic_tac_toes_t::action_t &action)
  {
    using action_t = tic_tac_toes_t::action_t;
    
    if (action == action_t::fill_1) {
      os << '1';
    }
    else if (action == action_t::fill_2) {
      os << '2';
    }
    else if (action == action_t::fill_3) {
      os << '3';
    }
    else if (action == action_t::fill_4) {
      os << '4';
    }
    else if (action == action_t::fill_5) {
      os << '5';
    }
    else if (action == action_t::fill_6) {
      os << '6';
    }
    else if (action == action_t::fill_7) {
      os << '7';
    }
    else if (action == action_t::fill_8) {
      os << '8';
    }
    else if (action == action_t::fill_9) {
      os << '9';
    }
    else if (action == action_t::NextRound) {
        os << '/';
    }
    else {
      os << '?';
      getchar();
    }
    
    return os;
  }

  auto tic_tac_toes_t::infoset_t::str() const -> std::string {
    stringstream ss;

    if (!history.empty()) {
        ss << "/";
    }

    for (const auto& a : history) {
        ss << a;
    }

    return ss.str();
}
      
      
  auto tic_tac_toes_t::str() const -> std::string {
      stringstream ss;

      if (!history().empty()) {
        ss << "/";
      }

      for (const auto& a : history()) {
        ss << a;
      }

      return ss.str();
}


  bool tic_tac_toes_t::infoset_t::is_equal(const infoset_t::concept_t &that) const {
    if (typeid(*this) == typeid(that)) {
      auto that_ = static_cast<const tic_tac_toes_t::infoset_t &>(that);
      
      return player == that_.player &&
             pieces_P1 == that_.pieces_P1 &&
             pieces_P2 == that_.pieces_P2 &&
             history == that_.history;
    }
    else {
      
      return false;
    }
    
    
  }

  size_t tic_tac_toes_t::infoset_t::hash() const {
    size_t seed = 0;
    hash_combine(seed, player);
    for (const auto &a : pieces_P1) {
      hash_combine(seed, a);
    }
    for (const auto &a : pieces_P2) {
      hash_combine(seed, a);
    }
    for (const auto &a : history) { hash_combine(seed, a); }
    
    return seed;
  }
  

} // namespace oz
