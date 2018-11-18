#ifndef OZ_LEDUC_TARGET_H
#define OZ_LEDUC_TARGET_H

#include <set>

#include "target.h"
#include "games/leduc.h"

namespace oz {

class leduc_target_t final : public target_t::concept_t {
 public:
  set<action_t> target_actions(const infoset_t &target_infoset,
                               const history_t &current_history) const override;
};

} // namespace oz

#endif // OZ_LEDUC_TARGET_H
