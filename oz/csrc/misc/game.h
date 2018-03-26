#ifndef OZ_GAME_H
#define OZ_GAME_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using real_t = double;
using prob_t = double;
using value_t = double;

enum class player_t {
  Chance = 0,
  P1 = 1,
  P2 = 2
};

class action_t {

};

class infoset_t {
 public:
  std::string str() const;
  std::vector<Action> actions() const;
};

class history_t {
 public:
  void act(action_t a);
  history_t operator >>(action_t a);
  infoset_t infoset(player_t p) const;
  player_t player() const;
  bool is_terminal() const;
  value_t utility() const;
};

#endif // OZ_GAME_H
