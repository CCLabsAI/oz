#include <catch.hpp>

#include "target.h"
#include "target/leduk_target.h"

#include "oos.h"

using namespace std;
using namespace oz;

TEST_CASE("targeting leduk histories", "[target]") {
  auto targeter = make_target<leduk_target_t>();
  auto& leduk_targeter = targeter.cast<leduk_target_t>();

  auto h = make_history<leduk_poker_t>();
  auto &target_game = leduk_targeter.target_game;

  const auto targets = targeter.target_actions(h);

  CHECK(targets.empty());

  h.act(make_action(leduk_poker_t::action_t::J1));
  h.act(make_action(leduk_poker_t::action_t::J2));

  target_game.act(make_action(leduk_poker_t::action_t::J1));
  target_game.act(make_action(leduk_poker_t::action_t::J2));
  target_game.act(make_action(leduk_poker_t::action_t::Raise));

  const auto targets2 = targeter.target_actions(h);

  CHECK(targets2.size() == 1);
  CHECK(*begin(targets2) == make_action(leduk_poker_t::action_t::Raise));

  h.act(make_action(leduk_poker_t::action_t::Raise));
  target_game.act(make_action(leduk_poker_t::action_t::Call));

  const auto targets3 = targeter.target_actions(h);

  CHECK(targets3.size() == 1);
  CHECK(*begin(targets3) == make_action(leduk_poker_t::action_t::Call));

  h.act(make_action(leduk_poker_t::action_t::Call));
  target_game.act(make_action(leduk_poker_t::action_t::K));

  const auto targets4 = targeter.target_actions(h);

  CHECK(targets4.size() == 1);
  CHECK(*begin(targets4) == make_action(leduk_poker_t::action_t::K));
}

TEST_CASE("targeting oos search", "[target]") {
  auto h = make_history<leduk_poker_t>();
  target_t target = make_target<leduk_target_t>();
  oos_t s;
  tree_t tree_targeted;
  tree_t tree_untargeted;
  rng_t rng(1);

  auto &game = target.cast<leduk_target_t>().target_game;
  game.act(make_action(leduk_poker_t::action_t::J1));
  game.act(make_action(leduk_poker_t::action_t::J2));
  game.act(make_action(leduk_poker_t::action_t::Raise));
  game.act(make_action(leduk_poker_t::action_t::Call));
  game.act(make_action(leduk_poker_t::action_t::K));

  s.search(h, 2000, tree_targeted, rng, target);
  s.search(h, 2000, tree_untargeted, rng);

  const auto infoset = game.infoset();
  const auto &node_targeted = tree_targeted.lookup(infoset);
  const auto &node_untargeted = tree_untargeted.lookup(infoset);

  const auto targeted_updates = node_targeted.regret_n();
  const auto untargeted_updates = node_untargeted.regret_n();

  CHECK(targeted_updates > untargeted_updates);

  tree_t tree_targeted_strong;
  s.search(h, 2000, tree_targeted_strong, rng, target, 0.1, 0.9, 0.01);

  const auto &node_strong = tree_targeted_strong.lookup(infoset);
  const auto targeted_strong_updates = node_strong.regret_n();

  CHECK(targeted_strong_updates > targeted_updates);
}
