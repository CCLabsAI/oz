#ifndef OZ_GOOFSPIEL2_TARGET_H
#define OZ_GOOFSPIEL2_TARGET_H

#include "target.h"

#include "games/goofspiel2.h"

namespace oz {

class goofspiel2_target_t final : public target_t::concept_t {
 public:
  goofspiel2_target_t(player_t match_player, int n) :
      match_player(match_player),
      target_game(n) { }

  set<action_t> target_actions(const history_t &current_history) const override;
  game_t &game() override;

  player_t match_player;
  goofspiel2_t target_game;
};

} // namespace oz

#endif // OZ_GOOFSPIEL2_TARGET_H
