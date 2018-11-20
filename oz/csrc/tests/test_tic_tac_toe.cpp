#include <catch.hpp>

#include "game.h"
#include "games/tic_tac_toes.h"


#include "target.h"
#include "target/tic_tac_toes_target.h"

#include "oos.h"
#include "tree.h"

using namespace std;
using namespace oz;

TEST_CASE("Tic Tac Toe basic actions", "[TicTacToe]") {
  auto game = tic_tac_toes_t();

  CHECK(!game.is_terminal());

  CHECK(game.player() == P1);

  game.act(make_action(0));
  CHECK(game.player() == P2);

  game.act(make_action(1));
  CHECK(game.player() == P1);

  game.act(make_action(4));
  game.act(make_action(2));
  
  game.act(make_action(8));

  CHECK(game.is_terminal());
}

TEST_CASE("Tic Tac Toe same player after an illegal move", "[TicTacToe]") {
  auto game = tic_tac_toes_t();

  CHECK(!game.is_terminal());

  CHECK(game.player() == P1);

  game.act(make_action(0));
  CHECK(game.player() == P2);

  game.act(make_action(1));
  CHECK(game.player() == P1);
    
  game.act(make_action(1));
  CHECK(game.player() == P1);
}

TEST_CASE("tic_tac_toe utility P1 wins", "[tic_tac_toe]") {
  auto game = tic_tac_toes_t();

  game.act(make_action(0));
  game.act(make_action(4));

  game.act(make_action(1));
  game.act(make_action(5));
    
  game.act(make_action(2));

  CHECK(game.is_terminal());
  CHECK(game.utility(P1) == 1);
}

TEST_CASE("tic_tac_toe utility P2 wins", "[tic_tac_toe]") {
  auto game = tic_tac_toes_t();

  game.act(make_action(0));
  game.act(make_action(3));

  game.act(make_action(8));
  game.act(make_action(4));
    
  game.act(make_action(7));
  game.act(make_action(5));

  CHECK(game.is_terminal());
  CHECK(game.utility(P2) == 1);
}

TEST_CASE("tic_tac_toe utility draw", "[tic_tac_toe]") {
  auto game = tic_tac_toes_t();

  game.act(make_action(0));
  game.act(make_action(2));

  game.act(make_action(1));
  game.act(make_action(3));
    
  game.act(make_action(5));
  game.act(make_action(4));
    
  
  game.act(make_action(6));
  game.act(make_action(7));
    
  
  game.act(make_action(8));

  CHECK(game.is_terminal());
    
  CHECK(game.utility(P1) == 0);
  CHECK(game.utility(P2) == 0);
}


TEST_CASE("tic tac toe pieces_P1 and pieces_P2 first element", "[tic tac toe]") {
  
  using action_vector_t = tic_tac_toes_t::action_vector_t;

  auto game = tic_tac_toes_t();
  game.act(make_action(1));
  
  // Now in the history of P1 there will be the legal move and the NextTurn action
  CHECK(game.pieces_P1()[0] == tic_tac_toes_t::action_t::fill_2);
  
  game.act(make_action(2));
  CHECK(game.pieces_P2()[0] == tic_tac_toes_t::action_t::fill_3);
  
  
}


TEST_CASE("tic tac toe pieces_P1 and pieces_P2", "[tic tac toe]") {
  
  using action_vector_t = tic_tac_toes_t::action_vector_t;

  auto game = tic_tac_toes_t();

  CHECK(game.pieces_P1().size() == 0);
  CHECK(game.pieces_P2().size() == 0);
    
  game.act(make_action(1));
  
  // Now in the history of P1 there will be the legal move and the NextTurn action
  CHECK(game.pieces_P1().size() == 2);
  
  game.act(make_action(2));
  CHECK(game.pieces_P2().size() == 2);
  
  // Illegal action
  game.act(make_action(2));
  // Legal action
  game.act(make_action(3));
  CHECK(game.pieces_P1().size() == 5);
  
  // Legal action for P2
  game.act(make_action(0));
  CHECK(game.pieces_P2().size() == 4);
  
}
