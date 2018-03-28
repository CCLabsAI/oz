#ifndef OZ_GAME_H
#define OZ_GAME_H

#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace oz {

using real_t = double;
using prob_t = double;
using value_t = double;

enum class player_t {
  Chance = 0,
  P1 = 1,
  P2 = 2
};

constexpr player_t CHANCE = player_t::Chance;
constexpr player_t P1 = player_t::P1;
constexpr player_t P2 = player_t::P2;

class action_t {
 public:
  action_t() : index_(-1) {};
  explicit action_t(int index) : index_(index) {};

  int index() const { return index_; };

 private:
  int index_;
};

class infoset_t {
 public:
  struct concept_t {
    virtual std::vector<action_t> actions() const = 0;
    virtual std::string str() const = 0;
    virtual bool is_equal(const concept_t& that) const = 0;
    virtual size_t hash() const = 0;
    virtual ~concept_t() = default;
  };

  std::vector<action_t> actions() const { return self_->actions(); }
  std::string str() const { return self_->str(); }
  virtual bool is_equal(const infoset_t& that) const { return self_->is_equal(*that.self_); };
  virtual size_t hash() const { return self_->hash(); };

 private:
  explicit infoset_t(concept_t *self) : self_(self) {};
  template<class Infoset, typename... Args>
  friend infoset_t make_infoset(Args&& ... args);

  std::shared_ptr<const concept_t> self_;
};

template<class Infoset, typename... Args>
auto make_infoset(Args&& ... args) -> infoset_t {
  return infoset_t(new Infoset(std::forward<Args>(args)...));
}

class game_t {
 public:
  virtual void act(action_t a) = 0;
  virtual infoset_t infoset() const = 0;
  virtual player_t player() const = 0;
  virtual bool is_terminal() const = 0;
  virtual value_t utility(player_t player) const = 0;
  virtual std::unique_ptr<game_t> clone() const = 0;
  virtual ~game_t() = default;
};

inline bool operator ==(const infoset_t& a, const infoset_t& b) {
  return a.is_equal(b);
}

inline bool operator !=(const infoset_t& a, const infoset_t& b) {
  return !(a == b);
}

inline bool operator ==(const action_t& a, const action_t& b) {
  return a.index() == b.index();
}

inline bool operator !=(const action_t& a, const action_t& b) {
  return !(a == b);
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
  inline size_t operator ()(const oz::infoset_t& a) const {
    return a.hash();
  }
};

} // namespace std

#endif // OZ_GAME_H
