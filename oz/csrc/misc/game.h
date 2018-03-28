#ifndef OZ_GAME_MISC_H
#define OZ_GAME_MISC_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using real_t = double;
using prob_t = double;
using value_t = double;

enum class player_t {
  Chance = 0,
  P1 = 1,
  P2 = 2
};

class action_t {

};

class infoset_t {
 public:
  std::string str() const;
  std::vector<action_t> actions() const;
};

bool operator ==(const infoset_t& a, const infoset_t& b);
bool operator ==(const action_t& a, const action_t& b);

class history_t {
 public:
  void act(action_t a) { self_->act(a); }
  infoset_t infoset() const { return self_->infoset(); }
  player_t player() const { return self_->player(); }
  bool is_terminal() const { return self_->is_terminal(); }
  value_t utility() const { return self_->utility(); }
  history_t operator >>(action_t a) {
    history_t h = std::move(*this);
    h.act(a);
    return h;
  }

 private:
  struct concept_t {
    virtual void act(action_t a) = 0;
    virtual infoset_t infoset() const = 0;
    virtual player_t player() const = 0;
    virtual bool is_terminal() const = 0;
    virtual value_t utility() const = 0;
  };

  // FIXME
  std::unique_ptr<concept_t> self_;
  // concept_t *self_;
};

#endif // OZ_GAME_H
