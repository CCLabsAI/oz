#ifndef OZ_TARGET_H
#define OZ_TARGET_H

#include <set>

#include "oos.h"

namespace oz {

using std::set;

class target_t {
  virtual set<action_t> target_actions(history_t h) = 0;
};

}

#endif // OZ_TARGET_H
