#ifndef OZ_HISTORY_H
#define OZ_HISTORY_H

#include "oos_base.h"

namespace oz {

class history_t final {
 public:
  using action_prob_map_t = game_t::action_prob_map_t;

  history_t(const history_t& that) : self_(that.self_->clone()) {};
  history_t(history_t&& that) = default;

  history_t &operator=(history_t &&that) = default;

  void act(action_t a) { self_->act(a); }
  infoset_t infoset() const { return self_->infoset(); }
  player_t player() const { return self_->player(); }
  bool is_terminal() const { return self_->is_terminal(); }
  value_t utility(player_t player) const { return self_->utility(player); }

  action_prob_map_t chance_actions() const
    { return self_->chance_actions(); }

  infoset_t infoset(infoset_t::allocator_t alloc) const
    { return self_->infoset(alloc); }

  action_prob_map_t chance_actions(game_t::action_prob_allocator_t alloc) const
    { return self_->chance_actions(alloc); }

  string str() const { return self_->str(); }

  history_t operator >>(action_t a) const {
    auto g = self_->clone();
    g->act(a);
    return history_t(move(g));
  }

  action_prob_t sample_chance(rng_t &rng) const;

  template <class T>
  const T &cast() const { return assert_cast<const T&>(*self_); }

 private:
  using ptr_t = std::unique_ptr<game_t>;

  explicit history_t(ptr_t self) : self_(move(self)) {};

  template<class Infoset, typename... Args>
  friend history_t make_history(Args&& ... args);

  ptr_t self_;
};

template<class Game, typename... Args>
auto make_history(Args&& ... args) -> history_t {
  return history_t(std::make_unique<Game>(std::forward<Args>(args)...));
}

action_prob_t sample_chance(const history_t &history, rng_t& rng,
                            game_t::action_prob_allocator_t alloc);

action_prob_t sample_action(const history_t &h, rng_t &rng);

} // namespace oz

#endif // OZ_HISTORY_H
