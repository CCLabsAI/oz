#include <catch.hpp>

#include <random>
#include <iostream>

#include "oos.h"
#include "best_response.h"

#include "games/flipguess.h"
#include "games/kuhn.h"

using namespace std;
using namespace oz;

TEST_CASE("oss simple", "[oss]") {
  auto h = make_history<kuhn_poker_t>();
  oos_t::search_t s(h, P1);
  tree_t tree;
  rng_t rng(1);

  s.select(tree, rng);
  REQUIRE(s.state() == oos_t::search_t::state_t::CREATE);
  s.create(tree, rng);
  CHECK(tree.size() == 1);
  REQUIRE(s.state() == oos_t::search_t::state_t::PLAYOUT);
}

TEST_CASE("node update", "[oss]") {
  auto heads = make_action(flipguess_t::action_t::Left);
  auto tails = make_action(flipguess_t::action_t::Right);
  auto actions = vector<action_t> { heads, tails };

  auto node = node_t(actions);

  node.regret(heads) += 7;
  CHECK(node.regret(heads) == 7);
  CHECK(node.regret(tails) == 0);

  node.average_strategy(tails) += 5;
  CHECK(node.average_strategy(tails) == 5);
  CHECK(node.average_strategy(heads) == 0);

}

TEST_CASE("tree update", "[oss]") {
  auto tree = tree_t();
  auto h = make_history<flipguess_t>();
  h.act(make_action(flipguess_t::action_t::Tails));
  auto infoset = h.infoset();
  auto actions = infoset.actions();
  auto heads = actions[0];
  auto tails = actions[1];

  tree.create_node(infoset);

  auto &node = tree.lookup(infoset);
  node.accumulate_regret(heads, 5);

  auto &node2 = tree.lookup(infoset);
  CHECK(node2.regret(heads) == 5);
  CHECK(node2.regret(tails) == 0);
}

TEST_CASE("oss playout", "[oss]") {
  auto h = make_history<kuhn_poker_t>();
  oos_t::search_t s(h, P1);
  tree_t tree;
  rng_t rng(1);

  s.select(tree, rng);
  REQUIRE(s.state() == oos_t::search_t::state_t::CREATE);
  s.create(tree, rng);
  REQUIRE(s.state() == oos_t::search_t::state_t::PLAYOUT);
  auto actions = s.infoset().actions();
  auto a = actions[0];
  s.playout_step(action_prob_t{ a, 1, 1, 1 });
  REQUIRE(s.state() == oos_t::search_t::state_t::BACKPROP);
  s.backprop(tree);
}

TEST_CASE("oss search", "[oss]") {
  auto h = make_history<flipguess_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);

  s.search(h, 25000, tree, rng);
  CHECK(tree.size() == 2);

  static const auto left = make_action(flipguess_t::action_t::Left);
  static const auto right = make_action(flipguess_t::action_t::Right);

  const auto node = tree.lookup(make_infoset<flipguess_t::infoset_t>(P2));
  const auto nl = node.average_strategy(left);
  const auto nr = node.average_strategy(right);
  CHECK(nl / (nl + nr) == Approx((prob_t) 1/3).epsilon(0.05));
}

TEST_CASE("oss exploitability flipguess", "[oss]") {
  auto h = make_history<flipguess_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);

  s.search(h, 100, tree, rng);
  auto sigma1 = tree.sigma_average();
  auto ex1 = exploitability(h, sigma1);

  s.search(h, 1000, tree, rng);
  auto sigma2 = tree.sigma_average();
  auto ex2 = exploitability(h, sigma2);

  CHECK(ex2 < ex1);
}

TEST_CASE("oss exploitability kuhn poker", "[oss]") {
  auto h = make_history<kuhn_poker_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  value_t ex = numeric_limits<value_t>::max();

  for(int i = 0; i < 5; ++i) {
    s.search(h, 5000, tree, rng);
    auto sigma = tree.sigma_average();
    auto ex_iter = exploitability(h, sigma);

    CHECK(ex_iter/ex < 1.5);
    ex = ex_iter;
  }
}
