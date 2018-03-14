#ifndef OZ_GAME_H
#define OZ_GAME_H

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

class Game {
  virtual reward_t reward() = 0;
  virtual Player player() = 0;
};

struct Infoset {

};

struct Action {

};

class History {
 public:
  Infoset infoset(Player p) const;
  Player player() const;
  History act(Action a);
  bool is_terminal() const;
  value_t utility() const;
};

#endif //OZ_GAME_H
