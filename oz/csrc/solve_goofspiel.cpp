#include "oos.h"
#include "best_response.h"
#include "games/goofspiel2.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<goofspiel2_t>(6);
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  value_t ex;

  cout << fixed << setprecision(3);

  for(int i = 0; i < 1000; i++) {
    s.search(h, 1000, tree, rng);
    cout << '.' << flush;
  }
  cout << endl;

  auto sigma = tree.sigma_average();
  ex = exploitability(h, sigma);
  cout << ex / 2;

  cout << endl;
}
