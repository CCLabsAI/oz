#include <catch.hpp>

#include "game.h"
#include "games/holdem.h"

using oz::holdem_poker_t;
using oz::CHANCE;
using oz::P1;
using oz::P2;

using card_t = holdem_poker_t::card_t;
using action_t = holdem_poker_t::action_t;

TEST_CASE("holdem poker basic actions", "[holdem]") {
  auto game = holdem_poker_t();
}
