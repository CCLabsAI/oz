#include "oss.h"

Res OSS::walk(History h, prob_t pi_i, prob_t pi_o, prob_t s1, prob_t s2, Player i)
{
  if (h.is_terminal()) {
    return {1, delta_*s1 + (1 - delta_)*s2, h.utility()};
  }
  else if (h.player() == Player::Chance) {
    Action a; // FIXME
    prob_t rho1;
    prob_t rho2;

    return walk(h.act(a), pi_i, rho2*pi_o, rho1*s1, rho2*s2, i);
  }

  Infoset I = h.infoset(h.player());

  return {1, 1, 0};
}
