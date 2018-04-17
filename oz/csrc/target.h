#ifndef OZ_TARGET_H
#define OZ_TARGET_H

#include <set>

#include "games/leduk.h"

namespace oz {

class history_t;

using std::vector;
using std::set;

class target_t final {
 public:
  target_t() = default;

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

  explicit target_t(ptr_t self): self_(move(self)) { }

  template<class T, typename... Args>
  friend target_t make_target(Args&& ... args);

  ptr_t self_;
};

template<class T, typename... Args>
target_t make_target(Args&& ... args) {
  return target_t(std::make_shared<T>(std::forward<Args>(args)...));
}


class leduk_target_t final : public target_t::concept_t {
 public:
  set<action_t> target_actions(const history_t &current_history) const override;

  leduk_poker_t target_game;

 private:
  static inline const leduk_poker_t &cast_history(const history_t &h);
};

} // namespace oz

#endif // OZ_TARGET_H
