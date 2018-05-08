#ifndef OZ_MCTS_H
#define OZ_MCTS_H

#include "game.h"

namespace oz {

class tree_t;

class mcts_search_t final {

  void select(const tree_t& tree, rng_t &rng); // walk from tip to leaf and updating path
  void create(tree_t& tree, rng_t &rng);       // add node to tree with zero values
  void playout_step(action_t a);               // step playout forward one ply
  void backprop(tree_t& tree);                 // unwind updates along path

  enum class state_t {
    SELECT,   // initial state
    CREATE,   // create node (with prior information)
    PLAYOUT,  // waiting for playout policy evaluation
    BACKPROP, // history is terminal, waiting to apply updates
    FINISHED
  };

  state_t state_;
  history_t history_;
  path_t path_;
  player_t search_player_;

}; // class search_t

} // namespace oz

#endif // OZ_MCTS_H
