#include "mcts.h"
#include "best_response.h"
#include "games/leduk.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<leduk_poker_t>();
  mcts::tree_t tree;
  rng_t rng(1);
  value_t ex;

  mcts::params_t params = {
      .c = 18,
      .nu = 0.9,
      .gamma = .1,
      .d = 0.002,
      .smooth = false,
      .search_player = P1
  };

  cout << fixed << setprecision(3);

  for (int i = 0; i < 100; i++) {
    mcts::search(h, 10000, tree, params, rng);
    auto sigma = tree.sigma_average();
    auto b1 = gebr(h, P1, sigma);
    auto b2 = gebr(h, P2, sigma);
    ex = b1 + b2;

    cout << '\r' << ex
         << ", " << b1
         << ", " << b2
         << flush;
  }

  cout << endl;

  for (const auto &p : tree.nodes) {
    const infoset_t &infoset = p.first;
    const mcts::node_t &node = p.second;
    cout << infoset.str()
         << " (N = " << node.n << ")"
         << endl;

    for (const auto q : node.q) {
      const action_t a = q.first;
      const mcts::q_val_t val = q.second;
      cout << '\t' << a.index()
           << ": " << val.w
           << ", " << val.n
           << " (" << val.v_uct(node.n, params.c) << ")"
           << endl;
    }
  }

  cout << "exploitability: " << ex
       << endl;
}
