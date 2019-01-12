#ifndef OZ_TIC_TAC_TOES_TARGET_H
#define OZ_TIC_TAC_TOES_TARGET_H

#include <set>

#include "target.h"
#include "games/tic_tac_toe.h"

namespace oz {

class tic_tac_toe_target_t final : public target_t::concept_t {
 public:
  set<action_t> target_actions(const infoset_t &target_infoset,
                               const history_t &current_history) const override;
};

} // namespace oz

#endif // OZ_TIC_TAC_TOES_TARGET_H
