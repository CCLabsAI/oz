#include "oos.h"
#include "best_response.h"
#include "games/leduk.h"
#include "games/goofspiel2.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  // history_t h = make_history<leduk_poker_t>();
  history_t h = make_history<goofspiel2_t>(6);
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  value_t ex;

  cout << fixed << setprecision(3);

  // do {
  //   s.search(h, 10000, tree, rng);
  //   auto sigma = tree.sigma_average();
  //   ex = exploitability(h, sigma);

  //   cout << '\r' << ex << flush;
  // } while(ex > 0.1);

  for(int i = 0; i < 1000; i++) {
    s.search(h, 1000, tree, rng);
    cout << '.' << flush;
  }
  count << endl;

  auto sigma = tree.sigma_average();
  ex = exploitability(h, sigma);
  cout << ex;

  cout << endl;
}
