#ifndef OZ_FLIPGUESS_H
#define OZ_FLIPGUESS_H

#include "game.h"

namespace oz {

using std::string;
using std::vector;

class flipguess_t final : public game_t {
 public:
  enum class action_t {
    NA = -1,
    Left = 1,
    Right = 2,
    Heads = 3,
    Tails = 4
  };

  class infoset_t : public oz::infoset_t::concept_t {
   public:
    explicit infoset_t(player_t player) : player_(player) {};

    vector<oz::action_t> actions() const override;
    string str() const override;
    bool is_equal(const oz::infoset_t::concept_t& that) const override;
    size_t hash() const override;

   private:
    player_t player_;
  };

  void act_(action_t a);

  void act(oz::action_t a) override { act_(a.as<action_t>()); };
  oz::infoset_t infoset() const override;
  player_t player() const override { return player_; }
  bool is_terminal() const override { return finished_; }
  value_t utility(player_t player) const override;

  std::unique_ptr<game_t> clone() const override {
    return std::make_unique<flipguess_t>(*this);
  }

  bool heads() { return heads_; };

 private:
  player_t player_ = CHANCE;
  bool finished_ = false;
  bool heads_ = false;
  action_t p1_action_ = action_t::NA;
  action_t p2_action_ = action_t::NA;
};

} // namespace oz

#endif // OZ_FLIPGUESS_H
