#include <catch.hpp>

#include <random>

#include "oss.h"
#include "games/flipguess.h"
#include "games/kuhn.h"

using namespace std;
using namespace oz;

TEST_CASE("oss simple", "[oss]") {
  auto h = make_history<kuhn_poker_t>();
  oss_t::search_t s(h, P1);
  tree_t tree;
  rng_t rng(1);

  s.select(tree, rng);
  REQUIRE(s.state() == oss_t::search_t::state_t::CREATE);
  s.create(tree, rng);
  REQUIRE(tree.size() == 1);
  REQUIRE(s.state() == oss_t::search_t::state_t::PLAYOUT);
}

TEST_CASE("oss playout", "[oss]") {
  auto h = make_history<kuhn_poker_t>();
  oss_t::search_t s(h, P1);
  tree_t tree;
  rng_t rng(1);

  s.select(tree, rng);
  REQUIRE(s.state() == oss_t::search_t::state_t::CREATE);
  s.create(tree, rng);
  REQUIRE(s.state() == oss_t::search_t::state_t::PLAYOUT);
  auto actions = s.infoset().actions();
  auto a = actions[0];
  s.playout_step(action_prob_t {a, 1, 1, 1});
  REQUIRE(s.state() == oss_t::search_t::state_t::BACKPROP);
  s.backprop(tree);
}
