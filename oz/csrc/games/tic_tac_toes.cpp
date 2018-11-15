#include "tic_tac_toes.h"
#include "hash.h"

#include <cassert>
#include <memory>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>

bool show_log = 0;
bool show_move = 0;

namespace oz {

  using namespace std;

  void tic_tac_toes_t::act_(action_t a) {
    
    unsigned int base_opponent_piece = 10;
    
    string discovery = "--";
    if (show_move == 1){
      cout << "Player " << player_idx(player_) <<endl;
    }
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
    
    cout << infoset().str() << endl;
    history_.push_back(a);
    // case current player is Player 1
    if (player_idx(player_) == 0){
        
        // Case the square chosen has been already chosen by the opponent in the past
        if (tot_moves_P2(0) == 1){
            tot_moves_P1(0) = 2;
            pieces_P1_.push_back(a);
        }
        // Legal move
        else{
            tot_moves_P1(0) = 1;
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
        if (tot_moves_P1(0) == 1){
            tot_moves_P2(0) = 2;
            pieces_P2_.push_back(a);
        }
        else{
            tot_moves_P2(0) = 1;
            pieces_P2_.push_back(a);
            pieces_P1_.push_back(action_t::NextRound);
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
    
    
    for (const auto& move : moves){
      if (move == action_t::fill_1){
        tot_moves[0] = 1;
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
    return make_infoset<infoset_t>(player_, pieces_P1_, pieces_P2_, history_, action_number, is_terminal_flag, tot_moves_P1_, tot_moves_P2_);

  }
  

  auto tic_tac_toes_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t
  {
    Expects(player() != CHANCE);
    
    return allocate_infoset<infoset_t, oz::infoset_t::allocator_t>
        (alloc,
         player_, pieces_P1_, pieces_P2_, history_,
         action_number, is_terminal_flag, tot_moves_P1_, tot_moves_P2_);
  }

  auto tic_tac_toes_t::is_terminal() const -> bool {
    return is_terminal_flag;
    
  }


  auto tic_tac_toes_t::infoset_t::actions() const -> actions_list_t {
    actions_list_t actions;
    
    
    if (player_idx(player) == 0){
      for (int i=0; i < 9; i++){
        if (tot_moves_P1[i] == 0){
          actions.push_back(make_action(i));
        }
      }
    }
        
      
    else{
      for (int i=0; i < 9; i++){
        if (tot_moves_P2[i] == 0){
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
             history == that_.history &&
             tot_moves_P1 == that_.tot_moves_P1 &&
             tot_moves_P2 == that_.tot_moves_P2;
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
    hash_combine(seed, tot_moves_P1[0]);
    hash_combine(seed, tot_moves_P1[1]);
    hash_combine(seed, tot_moves_P1[2]);
    hash_combine(seed, tot_moves_P1[3]);
    hash_combine(seed, tot_moves_P1[4]);
    hash_combine(seed, tot_moves_P1[5]);
    hash_combine(seed, tot_moves_P1[6]);
    hash_combine(seed, tot_moves_P1[7]);
    hash_combine(seed, tot_moves_P1[8]);

    hash_combine(seed, tot_moves_P2[0]);
    hash_combine(seed, tot_moves_P2[1]);
    hash_combine(seed, tot_moves_P2[2]);
    hash_combine(seed, tot_moves_P2[3]);
    hash_combine(seed, tot_moves_P2[4]);
    hash_combine(seed, tot_moves_P2[5]);
    hash_combine(seed, tot_moves_P2[6]);
    hash_combine(seed, tot_moves_P2[7]);
    hash_combine(seed, tot_moves_P2[8]);
    
    return seed;
  }
  

} // namespace oz
