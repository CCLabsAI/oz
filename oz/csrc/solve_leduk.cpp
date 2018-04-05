#include "oss.h"
#include "best_response.h"
#include "games/leduk.h"

#include <iostream>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<leduk_poker_t>();
  oss_t s;
  tree_t tree;
  rng_t rng(1);

  for(int i = 0; i < 10000; ++i) {
    s.search(h, 10000, tree, rng);
    auto sigma = tree.sigma_average();
    auto ex = exploitability(h, sigma);

    cout << ex << endl;
  }
}
