#include <set>

#include "best_response.h"

namespace oz {

using namespace std;

auto walk_infosets(history_t h, set<int> depths, int l) -> void {
  if (h.is_terminal()) {
    return;
  }

  const auto player = h.player();
  const auto infoset = h.infoset();

  if (player != Chance) {
    depths.insert(l);
  }

  for (const auto& a : infoset.actions()) {
    walk_infosets(h >> a, depths, l + 1);
  }
}

auto infoset_depths(history_t h) -> vector<int> {
  set<int> depths;
  walk_infosets(h, depths, 0);
  vector<int> depth_list(begin(depths), end(depths));
  sort(begin(depth_list), end(depth_list), greater<>());
  return depth_list;
}

}
