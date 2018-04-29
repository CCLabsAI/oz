#include <catch.hpp>

#include "game.h"
#include "games/goofspiel2.h"

using namespace oz;

TEST_CASE("goofspiel II basic actions", "[goofspiel2]") {
  auto game = goofspiel2_t(2);

  CHECK(!game.is_terminal());

  CHECK(game.player() == P1);

  game.act(make_action(0));
  CHECK(game.player() == P2);

  game.act(make_action(0));
  CHECK(game.player() == P1);

  game.act(make_action(1));
  game.act(make_action(1));

  CHECK(game.is_terminal());
}

TEST_CASE("goofspiel II utility", "[goofspiel2]") {
  auto game = goofspiel2_t(3);

  game.act(make_action(0));
  game.act(make_action(0));

  game.act(make_action(2));
  game.act(make_action(1));

  CHECK(game.score(P1) == 1);
  CHECK(game.score(P2) == 0);

  game.act(make_action(1));
  game.act(make_action(2));

  CHECK(game.score(P1) == 1);
  CHECK(game.score(P2) == 2);

  CHECK(game.is_terminal());
  CHECK(game.utility(P2) == 1);
}

static auto make_hand(set<goofspiel2_t::card_t> s) -> goofspiel2_t::hand_t {
  goofspiel2_t::hand_t hand;

  for(auto card : s) {
    hand.set(card);
  }

  return hand;
}

TEST_CASE("goofspiel II hand and bids", "[goofspiel2]") {
  using card_t = goofspiel2_t::card_t;
  using hand_t = goofspiel2_t::hand_t;
  using bids_t = goofspiel2_t::bids_t;
  using wins_t = goofspiel2_t::wins_t;

  auto game = goofspiel2_t(3);

  CHECK(game.hand(P1) == make_hand({0, 1, 2}));
  CHECK(game.hand(P2) == make_hand({0, 1, 2}));

  game.act(make_action(1));
  CHECK(game.hand(P1) == make_hand({0, 2}));
  game.act(make_action(2));

  game.act(make_action(2));
  game.act(make_action(1));
  CHECK(game.hand(P2) == make_hand({0}));

  CHECK(game.bids(P1) == bids_t {1, 2});
  CHECK(game.bids(P2) == bids_t {2, 1});

  game.act(make_action(0));
  game.act(make_action(0));
  CHECK(game.wins() == wins_t {P2, P1, CHANCE});
}
