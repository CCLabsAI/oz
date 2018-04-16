#include "oos.h"
#include "best_response.h"
#include "games/leduk.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<leduk_poker_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  value_t ex = numeric_limits<value_t>::infinity();

  cout << fixed << setprecision(3);

  while(ex > 0.1) {
    s.search(h, 10000, tree, rng);
    auto sigma = tree.sigma_average();
    ex = exploitability(h, sigma);

    cout << '\r' << ex << flush;
  }

  cout << endl;
}
