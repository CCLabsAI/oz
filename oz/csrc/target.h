#ifndef OZ_TARGET_H
#define OZ_TARGET_H

#include <set>

#include "game.h"

namespace oz {

class history_t;

using std::set;

class target_t final {
 public:
  struct concept_t {
    virtual set<action_t> target_actions(const history_t &h) const = 0;
  };

  set<action_t> target_actions(const history_t &h) const
  { return self_->target_actions(h); }

  explicit operator bool() const { return bool(self_); }

  template <class T>
  T &cast() { return assert_cast<T&>(*self_); }

 private:
  using ptr_t = std::shared_ptr<concept_t>;

  target_t() = default;
  explicit target_t(ptr_t self): self_(move(self)) { }

  template<class T, typename... Args>
  friend target_t make_target(Args&& ... args);
  friend target_t make_null_target();

  ptr_t self_;
};

template<class T, typename... Args>
target_t make_target(Args&& ... args) {
  return target_t(std::make_shared<T>(std::forward<Args>(args)...));
}

inline target_t make_null_target() {
  return target_t { };
}

} // namespace oz

#endif // OZ_TARGET_H
