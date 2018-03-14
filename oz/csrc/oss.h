#ifndef OZ_OSS_H
#define OZ_OSS_H

#include "game.h"

struct Res {
  prob_t x;
  prob_t l;
  value_t u;
};

class OSS {
 public:
  Res walk(History h, prob_t pi_i, prob_t pi_o, prob_t s1, prob_t s2, Player i);

 private:
  real_t delta_;
  tree_memory
};

#endif //OZ_OSS_H
