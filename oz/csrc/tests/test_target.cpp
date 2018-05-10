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

  auto h = make_history<leduk_poker_t>();
  auto h_target = make_history<leduk_poker_t>();

  h.act(make_action(action_t::J1));
  h.act(make_action(action_t::J2));

  h_target.act(make_action(action_t::J1));
  h_target.act(make_action(action_t::J2));

  const auto targets = targeter.target_actions(h_target.infoset(), h);
  CHECK(targets.empty());

  h_target.act(make_action(action_t::Raise));

  const auto targets2 = targeter.target_actions(h_target.infoset(), h);

  CHECK(targets2.size() == 1);
  CHECK(*begin(targets2) == make_action(leduk_poker_t::action_t::Raise));

  h.act(make_action(action_t::Raise));
  h_target.act(make_action(action_t::Call));
  h_target.act(make_action(action_t::K));

  const auto targets3 = targeter.target_actions(h_target.infoset(), h);

  CHECK(targets3.size() == 1);
  CHECK(*begin(targets3) == make_action(leduk_poker_t::action_t::Call));

  h.act(make_action(action_t::Call));

  const auto targets4 = targeter.target_actions(h_target.infoset(), h);

  CHECK(targets4.size() == 1);
  CHECK(*begin(targets4) == make_action(leduk_poker_t::action_t::K));
}

TEST_CASE("targeting goofspiel2 histories", "[target]") {
  auto targeter = make_target<goofspiel2_target_t>();
  auto& goof_targeter = targeter.cast<goofspiel2_target_t>();

  auto h = make_history<goofspiel2_t>(3);
  auto h_target = make_history<goofspiel2_t>(3);

  h_target.act(make_action(1));
  h_target.act(make_action(0));

  h_target.act(make_action(2));

  {
    const auto targets = targeter.target_actions(h_target.infoset(), h);
    CHECK(targets == set<action_t> { make_action(1), make_action(2) });
  }

  h.act(make_action(1));

  {
    const auto targets = targeter.target_actions(h_target.infoset(), h);
    CHECK(targets == set<action_t> { make_action(0) });
  }
}

TEST_CASE("targeting goofspiel2 regression 1", "[target]") {
  auto n_cards = 4;
  auto targeter = make_target<goofspiel2_target_t>();
  auto& goof_targeter = targeter.cast<goofspiel2_target_t>();

  auto h = make_history<goofspiel2_t>(n_cards);
  auto h_target = make_history<goofspiel2_t>(n_cards);

  h_target.act(make_action(3)); // P1
  h_target.act(make_action(0)); // P2

  h_target.act(make_action(2)); // P1
  h_target.act(make_action(3)); // P2

  h_target.act(make_action(1)); // P1
  h_target.act(make_action(2)); // P2

  h_target.act(make_action(0)); // P1

  CHECK(h_target.player() == P2);

  // target game
  // P1: 3 - 2 - 1 - 0
  //     >   <   <
  // P2: 0 - 3 - 2

  // current game
  // P1: 1 - (2) - (0)
  //       - (0) -  X
  //     >    <     <
  // P2: 0 - (3) - (2)

  h.act(make_action(1)); // P1
  h.act(make_action(0)); // P2

  const auto target = targeter.target_actions(h_target.infoset(), h);
  CHECK(target == set<action_t> { make_action(2) });
}

TEST_CASE("targeting oos search", "[target]") {
  using action_t = leduk_poker_t::action_t;

  auto h = make_history<leduk_poker_t>();
  auto h_target = make_history<leduk_poker_t>();
  target_t target = make_target<leduk_target_t>();

  oos_t s;
  tree_t tree_targeted;
  tree_t tree_untargeted;

  rng_t rng_targeted(1);
  rng_t rng_untargeted(1);

  h_target.act(make_action(action_t::J1));
  h_target.act(make_action(action_t::J2));
  h_target.act(make_action(action_t::Raise));
  h_target.act(make_action(action_t::Call));
  h_target.act(make_action(action_t::K));

  const auto &target_infoset = h_target.infoset();

  s.reset_targeting_ratio();
  s.search_targeted(h, 5000, tree_targeted, rng_targeted,
                    target, target_infoset,
                    0.1, 0.6, 0.01, .99);
  const auto tr_targeted = s.avg_targeting_ratio();

  s.reset_targeting_ratio();
  s.search(h, 5000, tree_untargeted, rng_untargeted);
  const auto tr_untargeted = s.avg_targeting_ratio();

  const auto infoset = h_target.infoset();
  const auto &node_targeted = tree_targeted.lookup(infoset);
  const auto &node_untargeted = tree_untargeted.lookup(infoset);

  const auto targeted_updates = node_targeted.regret_n();
  const auto untargeted_updates = node_untargeted.regret_n();

  CHECK(targeted_updates > untargeted_updates);
  CHECK(tr_targeted < tr_untargeted);

  tree_t tree_targeted_strong;

  s.reset_targeting_ratio();
  s.search_targeted(h, 5000, tree_targeted_strong, rng_targeted,
                    target, target_infoset,
                    0.1, 0.9, 0.01, .99);

  const auto tr_strong = s.avg_targeting_ratio();
  const auto &node_strong = tree_targeted_strong.lookup(infoset);
  const auto targeted_strong_updates = node_strong.regret_n();

  CHECK(targeted_strong_updates > targeted_updates);
  CHECK(tr_strong < tr_targeted);
}

TEST_CASE("targeting and exploitability", "[target][!hide]") {
  // TODO
}
