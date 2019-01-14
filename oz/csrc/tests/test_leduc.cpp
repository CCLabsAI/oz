#include <catch.hpp>

#include "game.h"
#include "games/leduc.h"

using oz::leduc_poker_t;
using oz::CHANCE;
using oz::P1;
using oz::P2;

using card_t = leduc_poker_t::card_t;
using action_t = leduc_poker_t::action_t;

TEST_CASE("leduc poker basic actions", "[leduc]") {
  leduc_poker_t g;

  CHECK(g.player() == CHANCE);
  g.act_(action_t::J1);
  CHECK(g.player() == CHANCE);
  g.act_(action_t::Q2);
  CHECK(g.player() == P1);
  g.act_(action_t::Raise);
  CHECK(g.player() == P2);
  g.act_(action_t::Call);
  CHECK(g.player() == CHANCE);
}

leduc_poker_t _leduc_after_deal() {
  auto g = leduc_poker_t();
  g.act_(action_t::J1);
  g.act_(action_t::Q2);
  return g;
}

TEST_CASE("leduc poker bet", "[leduc]") {
  leduc_poker_t g = _leduc_after_deal();

  g.act_(action_t::Raise);
  g.act_(action_t::Call);

  CHECK(g.pot(P1) == 3);
  CHECK(g.pot(P2) == 3);
  CHECK(g.round() == 1);
}

TEST_CASE("leduc poker check", "[leduc]") {
  leduc_poker_t g = _leduc_after_deal();

  g.act_(action_t::Call);
  g.act_(action_t::Call);

  CHECK(g.pot(P1) == 1);
  CHECK(g.pot(P2) == 1);
  CHECK(g.round() == 1);
  CHECK(g.board() == card_t::NA);
}

TEST_CASE("leduc poker fold", "[leduc]") {
  leduc_poker_t g = _leduc_after_deal();

  g.act_(action_t::Raise);
  g.act_(action_t::Fold);

  CHECK(g.pot(P1) == 3);
  CHECK(g.pot(P2) == 1);
  CHECK(g.is_terminal());
  CHECK(g.utility(P1) == 1);
}

TEST_CASE("leduc poker reraise", "[leduc]") {
  leduc_poker_t g = _leduc_after_deal();

  g.act_(action_t::Raise);
  g.act_(action_t::Raise);

  // TODO
}

TEST_CASE("leduc poker bet rounds", "[leduc]") {
  leduc_poker_t g = _leduc_after_deal();

  g.act_(action_t::Raise);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);
  CHECK(g.player() == CHANCE);
  g.act_(action_t::K);
  g.act_(action_t::Raise);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);

  CHECK(g.pot(P1) == 13);
  CHECK(g.pot(P2) == 13);
  CHECK(g.is_terminal());
}

TEST_CASE("leduc poker bet reward", "[leduc]") {
  leduc_poker_t g = leduc_poker_t();

  g.act_(action_t::K1);
  g.act_(action_t::Q2);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);
  g.act_(action_t::J);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);

  CHECK(g.hand(P1) == card_t::King);
  CHECK(g.hand(P2) == card_t::Queen);
  CHECK(g.board() == card_t::Jack);
  CHECK(g.is_terminal());
  CHECK(g.utility(P1) == 7);

  g = leduc_poker_t();

  g.act_(action_t::K1);
  g.act_(action_t::Q2);
  g.act_(action_t::Raise);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);
  g.act_(action_t::Q);
  g.act_(action_t::Raise);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);

  CHECK(g.hand(P1) == card_t::King);
  CHECK(g.hand(P2) == card_t::Queen);
  CHECK(g.board() == card_t::Queen);
  CHECK(g.is_terminal());
  CHECK(g.utility(P1) == -13);

  g = leduc_poker_t();

  g.act_(action_t::Q1);
  g.act_(action_t::Q2);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);
  g.act_(action_t::K);
  g.act_(action_t::Raise);
  g.act_(action_t::Call);

  CHECK(g.hand(P1) == card_t::Queen);
  CHECK(g.hand(P2) == card_t::Queen);
  CHECK(g.board() == card_t::King);
  CHECK(g.is_terminal());
  CHECK(g.utility(P1) == 0);
}
