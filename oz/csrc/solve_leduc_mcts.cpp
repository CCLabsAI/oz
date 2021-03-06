#include "mcts.h"
#include "best_response.h"
#include "games/leduc.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<leduc_poker_t>();
  mcts::tree_t tree;
  rng_t rng(1);
  value_t ex;

  mcts::params_t params;
  params.c = 18;
  params.eta = 0.9;
  params.gamma = 0.1;
  params.d = 0.002;
  // params.smooth = false;
  // params.search_player = P1;

  cout << fixed << setprecision(3);

  for (int i = 0; i < 10000; i++) {
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
