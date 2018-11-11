#include "oos.h"
#include "tree.h"
#include "best_response.h"
#include "games/leduc.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<leduc_poker_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  value_t ex;

  cout << fixed << setprecision(3);

  for(int i = 0; i < 100; i++) {
    s.search(h, 10000, tree, rng);
    auto sigma = tree.sigma_average();
    ex = exploitability(h, sigma);

    cout << '\r' << ex << flush;
  }

  cout << endl;
}
