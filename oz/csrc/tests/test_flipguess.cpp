#include <catch.hpp>

#include "game.h"
#include "games/flipguess.h"

using namespace oz;

TEST_CASE("flip guess basic actions", "[flipguess]") {
  flipguess_t game;

  REQUIRE(game.player() == CHANCE);
  REQUIRE(!game.is_terminal());

  game.act_(flipguess_t::action_t::Heads);
  REQUIRE(game.player() == P1);

  game.act_(flipguess_t::action_t::Left);
  REQUIRE(game.is_terminal());

  REQUIRE(game.utility(P1) == 1);
  REQUIRE(game.utility(P2) == -1);
}

TEST_CASE("flip guess basic actions 2", "[flipguess]") {
  flipguess_t game;

  game.act_(flipguess_t::action_t::Tails);
  REQUIRE(game.player() == P1);
  REQUIRE(game.infoset().str() == "P1");

  game.act_(flipguess_t::action_t::Left);
  REQUIRE(game.player() == P2);
  REQUIRE(game.infoset().str() == "P2");

  game.act_(flipguess_t::action_t::Right);
  REQUIRE(game.is_terminal());

  REQUIRE(game.utility(P1) == 0);
  REQUIRE(game.utility(P2) == 0);
}

TEST_CASE("flip guess basic actions 3", "[flipguess]") {
  flipguess_t game;

  game.act_(flipguess_t::action_t::Tails);
  REQUIRE(game.player() == P1);
  game.act_(flipguess_t::action_t::Left);
  REQUIRE(game.player() == P2);

  game.act_(flipguess_t::action_t::Left);
  REQUIRE(game.is_terminal());

  REQUIRE(game.utility(P1) == 3);
  REQUIRE(game.utility(P2) == -3);
}

TEST_CASE("flip guess basic actions 4", "[flipguess]") {
  flipguess_t game;

  game.act_(flipguess_t::action_t::Heads);
  REQUIRE(game.player() == P1);
  game.act_(flipguess_t::action_t::Right);
  REQUIRE(game.is_terminal());

  REQUIRE(game.utility(P1) == 0);
  REQUIRE(game.utility(P2) == 0);
}

TEST_CASE("flip guess action equality", "[flipguess]") {
  flipguess_t game;

  action_t a(static_cast<int>(flipguess_t::action_t::Heads));
  action_t b(static_cast<int>(flipguess_t::action_t::Heads));

  REQUIRE(a == b);

  action_t c(static_cast<int>(flipguess_t::action_t::Left));
  action_t d(static_cast<int>(flipguess_t::action_t::Right));

  REQUIRE(c != d);
}

TEST_CASE("flip guess infoset equality", "[flipguess]") {
  flipguess_t game;

  infoset_t a = make_infoset<flipguess_t::infoset_t>(P1);
  infoset_t b = make_infoset<flipguess_t::infoset_t>(P1);

  REQUIRE(a == b);

  infoset_t c = make_infoset<flipguess_t::infoset_t>(CHANCE);
  infoset_t d = make_infoset<flipguess_t::infoset_t>(P2);

  REQUIRE(c != d);
}
