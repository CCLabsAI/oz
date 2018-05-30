#ifndef OZ_LIARS_DICE_TARGET_H
#define OZ_LIARS_DICE_TARGET_H

#include <set>

#include "target.h"
#include "games/liars_dice.h"

namespace oz {

class liars_dice_target_t final : public target_t::concept_t {
 public:
  set<action_t> target_actions(const infoset_t &target_infoset,
                               const history_t &current_history) const override;
};

} // namespace oz

#endif // OZ_LIARS_DICE_TARGET_H
