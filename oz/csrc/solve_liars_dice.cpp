//
// Created by Michela on 23/5/18.
//

#include "oos.h"
#include "tree.h"
#include "best_response.h"
#include "games/liars_dice.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<liars_dice_t>();
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

