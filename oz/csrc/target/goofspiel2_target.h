#ifndef OZ_GOOFSPIEL2_TARGET_H
#define OZ_GOOFSPIEL2_TARGET_H

#include "target.h"

#include "games/goofspiel2.h"

namespace oz {

class goofspiel2_target_t final : public target_t::concept_t {
 public:
  set<action_t> target_actions(const infoset_t &target_infoset,
                               const history_t &current_history) const override;
};

} // namespace oz

#endif // OZ_GOOFSPIEL2_TARGET_H
