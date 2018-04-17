#ifndef OZ_TARGET_H
#define OZ_TARGET_H

#include <set>

#include "oos.h"
#include "games/leduk.h"

namespace oz {

using std::vector;
using std::set;

class target_t {
 public:
  virtual set<action_t> target_actions(const history_t &current_history,
                                       const history_t &target_history,
                                       vector<action_t> actions) const = 0;
};

class leduk_target_t final : public target_t {
 public:
  set<action_t> target_actions(const history_t &current_history,
                               const history_t &target_history,
                               vector<action_t> actions) const override;

 private:
  static inline const leduk_poker_t &cast_history(const history_t &h);
};

} // namespace oz

#endif // OZ_TARGET_H
