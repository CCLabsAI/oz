#include <catch.hpp>

#include "mcts.h"
#include "games/kuhn.h"

using namespace std;
using namespace oz;

TEST_CASE("mcts simple", "[mcts]") {
  history_t h = make_history<kuhn_poker_t>();
  mcts::tree_t tree;
  rng_t rng(1);

  mcts::search(h, 1000, tree, rng, 2);
}
