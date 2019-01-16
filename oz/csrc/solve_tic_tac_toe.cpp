//
// Created by Michela on 23/5/18.
//

#include "oos.h"
#include "tree.h"
#include "best_response.h"
#include "games/tic_tac_toe.h"

#include <limits>
#include <iostream>
#include <iomanip>

using namespace oz;
using namespace std;

int main(int argc, char **argv) {
  history_t h = make_history<tic_tac_toe_t>();
  oos_t s;
  tree_t tree;
  rng_t rng(1);
  for(int i = 0; i < 100; i++) {
    s.search(h, 10000, tree, rng);
    cout << '.' << flush;
    
  }
  cout << endl;
}

