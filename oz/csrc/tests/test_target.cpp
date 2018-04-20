#include <catch.hpp>

#include "target.h"
#include "target/leduk_target.h"
#include "target/goofspiel2_target.h"

#include "oos.h"

using namespace std;
using namespace oz;

TEST_CASE("targeting leduk histories", "[target]") {
  using action_t = leduk_poker_t::action_t;

  auto targeter = make_target<leduk_target_t>();
  auto& leduk_targeter = targeter.cast<leduk_target_t>();

  auto h = make_history<leduk_poker_t>();
  auto &target_game = leduk_targeter.target_game;

  const auto targets = targeter.target_actions(h);

  CHECK(targets.empty());

  h.act(make_action(action_t::J1));
  h.act(make_action(action_t::J2));

  target_game.act(make_action(action_t::J1));
  target_game.act(make_action(action_t::J2));
  target_game.act(make_action(action_t::Raise));

  const auto targets2 = targeter.target_actions(h);

  CHECK(targets2.size() == 1);
  CHECK(*begin(targets2) == make_action(leduk_poker_t::action_t::Raise));

  h.act(make_action(action_t::Raise));
  target_game.act(make_action(action_t::Call));

  const auto targets3 = targeter.target_actions(h);

  CHECK(targets3.size() == 1);
  CHECK(*begin(targets3) == make_action(leduk_poker_t::action_t::Call));

  h.act(make_action(action_t::Call));
  target_game.act(make_action(action_t::K));

  const auto targets4 = targeter.target_actions(h);

  CHECK(targets4.size() == 1);
  CHECK(*begin(targets4) == make_action(leduk_poker_t::action_t::K));
}

TEST_CASE("targeting goofspiel2 histories", "[target]") {
  auto targeter = make_target<goofspiel2_target_t>(P2, 3);
  auto& goof_targeter = targeter.cast<goofspiel2_target_t>();

  auto game = make_history<goofspiel2_t>(3);
  auto &target_game = goof_targeter.target_game;

  target_game.act(make_action(1));
  target_game.act(make_action(0));

  {
    const auto targets = targeter.target_actions(game);
    CHECK(targets == set<action_t> { make_action(1), make_action(2) });
  }

  game.act(make_action(1));

  {
    const auto targets = targeter.target_actions(game);
    CHECK(targets == set<action_t> { make_action(0) });
  }
}

TEST_CASE("targeting goofspiel2 regression 1", "[target]") {
  auto n_cards = 4;
  auto targeter = make_target<goofspiel2_target_t>(P2, n_cards);
  auto& goof_targeter = targeter.cast<goofspiel2_target_t>();

  auto game = make_history<goofspiel2_t>(n_cards);
  auto &target_game = goof_targeter.target_game;

  target_game.act(make_action(3)); // P1
  target_game.act(make_action(0)); // P2
  
  target_game.act(make_action(2)); // P1
  target_game.act(make_action(3)); // P2

  target_game.act(make_action(1)); // P1
  target_game.act(make_action(2)); // P2

  game.act(make_action(1)); // P1
  game.act(make_action(0)); // P2

  game.act(make_action(0)); // P1
  game.act(make_action(3)); // P2

  // this checks there is not an assertion error
  const auto targets = targeter.target_actions(game);
}

TEST_CASE("targeting oos search", "[target]") {
  using action_t = leduk_poker_t::action_t;

  auto h = make_history<leduk_poker_t>();
  target_t target = make_target<leduk_target_t>();
  oos_t s;
  tree_t tree_targeted;
  tree_t tree_untargeted;
  rng_t rng(1);

  auto &game = target.cast<leduk_target_t>().target_game;
  game.act(make_action(action_t::J1));
  game.act(make_action(action_t::J2));
  game.act(make_action(action_t::Raise));
  game.act(make_action(action_t::Call));
  game.act(make_action(action_t::K));

  s.retarget();
  s.search(h, 2000, tree_targeted, rng, target);
  const auto tr_targeted = s.avg_targeting_ratio();

  s.retarget();
  s.search(h, 2000, tree_untargeted, rng);
  const auto tr_untargeted = s.avg_targeting_ratio();

  const auto infoset = game.infoset();
  const auto &node_targeted = tree_targeted.lookup(infoset);
  const auto &node_untargeted = tree_untargeted.lookup(infoset);

  const auto targeted_updates = node_targeted.regret_n();
  const auto untargeted_updates = node_untargeted.regret_n();

  CHECK(targeted_updates > untargeted_updates);
  CHECK(tr_targeted < tr_untargeted);

  tree_t tree_targeted_strong;

  s.retarget();
  s.search(h, 2000, tree_targeted_strong, rng, target, 0.1, 0.9, 0.01);

  const auto tr_strong = s.avg_targeting_ratio();
  const auto &node_strong = tree_targeted_strong.lookup(infoset);
  const auto targeted_strong_updates = node_strong.regret_n();

  CHECK(targeted_strong_updates > targeted_updates);
  CHECK(tr_strong < tr_targeted);
}

TEST_CASE("targeting and exploitability", "[target][!hide]") {
  // TODO
}
