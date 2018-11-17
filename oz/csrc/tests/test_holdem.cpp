#include <catch.hpp>

#include "game.h"
#include "games/holdem.h"

using oz::holdem_poker_t;
using oz::CHANCE;
using oz::P1;
using oz::P2;

using card_t = holdem_poker_t::card_t;
using action_t = holdem_poker_t::action_t;


TEST_CASE("holdem poker deal utilities", "[holdem]") {
  for (card_t c1 = holdem_poker_t::CARD_MIN; c1 < holdem_poker_t::N_CARDS; c1++) {
    holdem_poker_t::action_t a = holdem_poker_t::deal_action_for_card(c1);
    card_t c2 = holdem_poker_t::card_for_deal_action(a);
    CHECK(holdem_poker_t::is_deal_action(a));
    CHECK(c1 == c2);
  }

  CHECK(!holdem_poker_t::is_deal_action(action_t::Raise));
  CHECK(!holdem_poker_t::is_deal_action(action_t::Call));
  CHECK(!holdem_poker_t::is_deal_action(action_t::Fold));
}

TEST_CASE("holdem poker basic actions", "[holdem]") {
  auto game = holdem_poker_t();

  while (game.player() == CHANCE) {
    auto action_probs = game.chance_actions();
    auto ap = *begin(action_probs);
    auto a = ap.first;

    game.act(a);
  }

  CHECK(game.hand(P1)[0] != holdem_poker_t::CARD_NA);
  CHECK(game.hand(P1)[1] != holdem_poker_t::CARD_NA);

  CHECK(game.hand(P2)[0] != holdem_poker_t::CARD_NA);
  CHECK(game.hand(P2)[1] != holdem_poker_t::CARD_NA);

  CHECK(game.player() == P2);
  CHECK(game.pot(P1) == 10);
  CHECK(game.pot(P2) == 5);
  game.act(make_action(action_t::Call));

  CHECK(game.player() == P1);
  game.act(make_action(action_t::Raise));
  CHECK(game.pot(P1) == 20);

  CHECK(game.player() == P2);
  game.act(make_action(action_t::Raise));
  CHECK(game.pot(P2) == 30);

  CHECK(game.player() == P1);
  game.act(make_action(action_t::Call));
  CHECK(game.player() == CHANCE);
  CHECK(game.round() == 1);

  CHECK(game.board().size() == 0);

  while (game.player() == CHANCE) {
    auto action_probs = game.chance_actions();
    auto ap = *begin(action_probs);
    auto a = ap.first;

    game.act(a);
  }

  CHECK(game.board().size() == 3);
}
