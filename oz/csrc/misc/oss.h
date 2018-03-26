#ifndef OZ_OSS_H
#define OZ_OSS_H

#include "game.h"

template<class History>
class OSS {
  struct walk_ret_t {
    prob_t x;
    prob_t l;
    value_t u;
  };

 public:
  walk_ret_t walk(History h,
                  prob_t pi_i, prob_t pi_o,
                  prob_t s1, prob_t s2, Player i);

 private:
  real_t delta_;
  real_t eps_;
};

template<class History>
auto OSS::walk(History h,
               prob_t pi_i, prob_t pi_o,
               prob_t s1, prob_t s2, Player i) -> walk_ret_t
{
  using Action = History::Action;
  using Infoset = History::Infoset;

  if (h.is_terminal()) {
    return {1, delta_*s1 + (1 - delta_)*s2, h.utility()};
  }
  else if (h.player() == Player::Chance) {
    Action a; // ...
    prob_t rho1 = 0; // ...
    prob_t rho2 = 0; // ...

    return walk(h.act(a), pi_i, rho2*pi_o, rho1*s1, rho2*s2, i);
  }

  Infoset I = h.infoset(h.player());

  return {1, 1, 0};
}

#endif // OZ_OSS_H
