#ifndef OZ_GAME_H
#define OZ_GAME_H

#include "util.h"

#include <memory>
#include <functional>
#include <random>
#include <string>
#include <map>

#include <boost/container/small_vector.hpp>
#include <boost/container/pmr/flat_map.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>

namespace oz {

using std::move;
using std::string;
using std::map;

using std::unique_ptr;
using std::shared_ptr;

using boost::container::pmr::polymorphic_allocator;
using boost::container::small_vector;

using real_t = double;
using prob_t = double;
using value_t = double;

using rng_t = std::mt19937;

enum class player_t {
  Chance = 0,
  P1 = 1,
  P2 = 2
};

constexpr player_t CHANCE = player_t::Chance;
constexpr player_t P1 = player_t::P1;
constexpr player_t P2 = player_t::P2;

class action_t final {
 public:
  static constexpr int UNK = -1000000;

  action_t() : index_(UNK) { };
  explicit action_t(int index) : index_(index) { };

  int index() const { return index_; };

  template <typename T>
  T cast() const { return static_cast<T>(index_); }

 private:
  int index_;
};

template <class Action>
action_t make_action(Action a) {
  return action_t(static_cast<int>(a));
}

class infoset_t final {
 public:
  static constexpr int N_ACTIONS_STATIC = 16;

  using actions_list_t = small_vector<action_t, N_ACTIONS_STATIC>;
  using allocator_t = polymorphic_allocator<infoset_t>;

  struct concept_t {
    using actions_list_t = infoset_t::actions_list_t;

    virtual actions_list_t actions() const = 0;
    virtual string str() const = 0;
    virtual bool is_equal(const concept_t& that) const = 0;
    virtual size_t hash() const = 0;
    virtual ~concept_t() = default;
  };

  actions_list_t actions() const { return self_->actions(); }
  string str() const { return self_->str(); }
  bool is_equal(const infoset_t& that) const { return self_->is_equal(*that.self_); };
  size_t hash() const { return self_->hash(); };

  explicit operator bool() const { return bool(self_); }

  template <class T>
  const T &cast() const { return assert_cast<const T&>(*self_); }

 private:
  using ptr_t = shared_ptr<const concept_t>;

  explicit infoset_t(ptr_t self) : self_(move(self)) { };

  template<class Infoset, typename... Args>
  friend infoset_t make_infoset(Args&& ... args);

  template<class Infoset, class Alloc, typename... Args>
  friend infoset_t allocate_infoset(const Alloc& alloc, Args&& ... args);

  friend infoset_t null_infoset();

  ptr_t self_;
};

template<class Infoset, typename... Args>
auto make_infoset(Args&& ... args) -> infoset_t {
  return infoset_t(std::make_shared<Infoset>(std::forward<Args>(args)...));
}

template<class Infoset, class Alloc, typename... Args>
auto allocate_infoset(const Alloc& alloc, Args&& ... args) -> infoset_t {
  return infoset_t(std::allocate_shared<Infoset>(alloc, std::forward<Args>(args)...));
}

inline auto null_infoset() -> infoset_t {
  return infoset_t(nullptr);
}

class game_t {
 public:
  using action_prob_allocator_t =
    polymorphic_allocator<std::pair<action_t, prob_t>>;
  using action_prob_map_t =
    boost::container::pmr::flat_map<action_t, prob_t>;

  virtual void act(action_t a) = 0;
  virtual infoset_t infoset() const = 0;
  virtual player_t player() const = 0;
  virtual bool is_terminal() const = 0;
  virtual value_t utility(player_t player) const = 0;
  virtual action_prob_map_t chance_actions() const = 0;

  virtual unique_ptr<game_t> clone() const = 0;
  virtual ~game_t() = default;

  virtual infoset_t infoset(infoset_t::allocator_t alloc) const
    { return infoset(); }

  virtual action_prob_map_t chance_actions(action_prob_allocator_t alloc) const
    { return chance_actions(); }

  virtual string str() const { throw std::logic_error("not implemented"); };
};


inline value_t relative_utility(player_t player, value_t u) {
  return player == P2 ? -u : u;
}

inline bool operator ==(const infoset_t &a, const infoset_t &b) {
  return a.is_equal(b);
}

inline bool operator !=(const infoset_t &a, const infoset_t &b) {
  return !(a == b);
}

inline bool operator ==(const action_t &a, const action_t &b) {
  return a.index() == b.index();
}

inline bool operator !=(const action_t &a, const action_t &b) {
  return !(a == b);
}

inline bool operator <(const action_t &a, const action_t &b) {
  return a.index() < b.index();
}

} // namespace oz

namespace std {

template<>
struct hash<oz::action_t> {
  inline size_t operator ()(const oz::action_t& a) const {
    return hash<int>()(a.index());
  }
};

template<>
struct hash<oz::infoset_t> {
  inline size_t operator ()(const oz::infoset_t& infoset) const {
    return infoset.hash();
  }
};

} // namespace std

#endif // OZ_GAME_H
