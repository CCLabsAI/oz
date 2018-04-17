#include <catch.hpp>

#include "target.h"

using namespace std;
using namespace oz;

// TODO move this into a utility module
template <typename T>
inline auto keys(const T &m) -> vector<typename T::key_type> {
  auto keys = vector<action_t>(m.size());

  transform(begin(m), end(m), begin(keys), [](const auto& p) {
    return p.first;
  });

  return keys;
}

// TODO move this into a utility module
static inline auto all_actions(const history_t& h) -> vector<action_t> {
  if (h.player() == CHANCE) {
    const auto actions_pr = h.chance_actions();
    return keys(actions_pr);
  }
  else {
    const auto infoset = h.infoset();
    return infoset.actions();
  }
}

TEST_CASE("targeting leduk histories", "[target]") {
  const auto targeter_ptr = unique_ptr<target_t>(new leduk_target_t());
  const auto& targeter = *targeter_ptr;

  auto h_current = make_history<leduk_poker_t>();
  auto h_target = make_history<leduk_poker_t>();

  const auto actions = all_actions(h_current);
  const auto targets = targeter.target_actions(h_current,
                                               h_target,
                                               actions);

  CHECK(equal(begin(targets), end(targets), begin(actions), end(actions)));

  h_current.act(make_action(leduk_poker_t::action_t::J1));
  h_current.act(make_action(leduk_poker_t::action_t::J2));

  h_target.act(make_action(leduk_poker_t::action_t::J1));
  h_target.act(make_action(leduk_poker_t::action_t::J2));
  h_target.act(make_action(leduk_poker_t::action_t::Raise));

  const auto targets2 = targeter.target_actions(h_current,
                                                h_target,
                                                all_actions(h_current));

  CHECK(targets2.size() == 1);
  CHECK(*begin(targets2) == make_action(leduk_poker_t::action_t::Raise));

  h_current.act(make_action(leduk_poker_t::action_t::Raise));
  h_target.act(make_action(leduk_poker_t::action_t::Call));

  const auto targets3 = targeter.target_actions(h_current,
                                                h_target,
                                                all_actions(h_current));

  CHECK(targets3.size() == 1);
  CHECK(*begin(targets3) == make_action(leduk_poker_t::action_t::Call));

  h_current.act(make_action(leduk_poker_t::action_t::Call));
  h_target.act(make_action(leduk_poker_t::action_t::K));

  const auto targets4 = targeter.target_actions(h_current,
                                                h_target,
                                                all_actions(h_current));

  CHECK(targets4.size() == 1);
  CHECK(*begin(targets4) == make_action(leduk_poker_t::action_t::K));
}
