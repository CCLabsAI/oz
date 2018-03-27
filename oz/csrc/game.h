#ifndef OZ_GAME_H
#define OZ_GAME_H

namespace oz {
  using real_t = double;
  using prob_t = double;
  using value_t = double;

  enum class player_t {
    Chance = 0,
    P1 = 1,
    P2 = 2
  };

  constexpr player_t Chance = player_t::Chance;
}


#endif // OZ_GAME_H
