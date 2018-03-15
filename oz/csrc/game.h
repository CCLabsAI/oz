#ifndef OZ_GAME_H
#define OZ_GAME_H

#include <cstdint>
#include <string>
#include <vector>

using real_t = double;
using prob_t = double;
using value_t = double;

enum class Player {
  Chance = 0,
  P1 = 1,
  P2 = 2
};

struct Infoset {

};

struct Action {
  intptr_t data;
};

class History {
 public:
  History act(Action a);
  Infoset infoset(Player p) const;
  Player player() const;
  bool is_terminal() const;
  value_t utility() const;
};

#endif //OZ_GAME_H
