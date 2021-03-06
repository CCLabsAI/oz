#include <catch.hpp>

#include "hash.h"
#include "best_response.h"
#include "games/flipguess.h"
#include "games/kuhn.h"

using namespace std;
using namespace oz;

class sigma_uniform_t final : public sigma_t::concept_t {
  prob_t pr(oz::infoset_t infoset, oz::action_t a) const override {
    auto n = infoset.actions().size();
    return (prob_t) 1 / n;
  }
};

class sigma_flip_t final : public sigma_t::concept_t {
 public:
  prob_t pr(oz::infoset_t infoset, oz::action_t a) const override {
    static const auto chance = make_infoset<flipguess_t::infoset_t>(CHANCE);
    static const auto p1 = make_infoset<flipguess_t::infoset_t>(P1);
    static const auto p2 = make_infoset<flipguess_t::infoset_t>(P2);

    static const auto left = make_action(flipguess_t::action_t::Left);
    static const auto right = make_action(flipguess_t::action_t::Right);

    if (infoset == p1) {
      return (prob_t) 1/2;
    }
    else if (infoset == p2) {
      if (a == left) {
        return (prob_t) 1 / 3;
      }
      else if (a == right) {
        return (prob_t) 2 / 3;
      }
      else {
        return 0;
      }
    }
    else {
      return (prob_t) 1 / infoset.actions().size();
    }
  }
};

class sigma_kuhn_t final : public sigma_t::concept_t {
 public:
  prob_t pr(oz::infoset_t infoset, oz::action_t a) const override {
    const auto &infoset_ = infoset.cast<kuhn_poker_t::infoset_t>();
    auto player = infoset_.player;

    static const auto bet = make_action(kuhn_poker_t::action_t::Bet);
    static const auto pass = make_action(kuhn_poker_t::action_t::Pass);

    if (player == P1 || player == P2) {
      if (a == bet) {
        return (prob_t) 1;
      }
      else if (a == pass) {
        return (prob_t) 0;
      }
      else {
        return (prob_t) 0;
      }
    }
    else {
      return (prob_t) 1 / infoset.actions().size();
    }
  }
};

TEST_CASE("best response infoset depths", "[best_response]") {
  auto h = make_history<flipguess_t>();
  auto d1 = infoset_depths(h, P1);
  auto d2 = infoset_depths(h, P2);

  REQUIRE(d1 == vector<int>{ 1 });
  REQUIRE(d2 == vector<int>{ 2 });
}

TEST_CASE("best response map lookup", "[best_response]") {
  auto h = make_history<flipguess_t>();
  h.act(make_action(flipguess_t::action_t::Heads));

  auto infoset = h.infoset();
  auto actions = infoset.actions();
  auto a1 = actions[0];
  auto a2 = actions[1];
  std::unordered_map<q_info_t, value_t> m;

  m[{ infoset, a1 }] += 1;
  m[{ infoset, a1 }] += 1;

  REQUIRE(m[{ infoset, a1 }] == 2);
  REQUIRE(m[{ infoset, a2 }] == 0);
}

TEST_CASE("best response run gebr_pass2", "[best_response]") {
  auto h = make_history<flipguess_t>();
  auto sigma = make_sigma<sigma_uniform_t>();

  q_stats_t tb;
  gebr_pass2(h, P1, 1, 0, 1.0, sigma, tb);

  REQUIRE(!tb.empty());
}

TEST_CASE("best response run gebr p1", "[best_response]") {
  auto h = make_history<flipguess_t>();
  auto sigma = make_sigma<sigma_uniform_t>();

  value_t v1 = gebr(h, P1, sigma);
  REQUIRE(v1 == 1.25);
}

TEST_CASE("best response run gebr p2", "[best_response]") {
  auto h = make_history<flipguess_t>();
  auto sigma = make_sigma<sigma_uniform_t>();

  value_t v2 = gebr(h, P2, sigma);
  REQUIRE(v2 == -1);
}

TEST_CASE("best response kuhn gebr", "[best_response]") {
  auto h = make_history<kuhn_poker_t>();
  auto sigma = make_sigma<sigma_kuhn_t>();

  value_t v2 = gebr(move(h), P2, sigma);
  REQUIRE(-v2 == (1./3)*1 + (1./3)*-2);
}

TEST_CASE("best response run exploitability", "[best_response]") {
  auto h = make_history<flipguess_t>();

  auto sigma_uniform = make_sigma<sigma_uniform_t>();
  auto ex_uniform = exploitability(h, sigma_uniform);
  REQUIRE(ex_uniform == 0.25);

  auto sigma_nash = make_sigma<sigma_flip_t>();
  auto v1 = gebr(h, P1, sigma_nash);
  auto v2 = gebr(h, P2, sigma_nash);
  REQUIRE(v1 == 1);
  REQUIRE(v2 == -1);

  value_t ex_nash = exploitability(h, sigma_nash);
  REQUIRE(ex_nash == 0);
}
