#ifndef OZ_LD_H
#define OZ_LD_H

#include <cassert>
#include <array>

#include "game.h"

namespace oz {

  using std::move;
  using std::string;
  using std::array;
  using std::vector;

  using boost::container::static_vector;

  class liar_dice_t final : public game_t {
  public:

    enum class action_t {

      // actions for P1 and P2 : Raising dice face or number of dice or call
          Raise_0face = 1,
      Raise_1face,
      Raise_2face,
      Raise_3face,
      Raise_4face,
      Raise_5face,

      Raise_0dice,
      Raise_1dice,
      Raise_2dice,
      Raise_3dice,
      Raise_4dice,
      Call_liar,

      NextRound = -10,

      // Actions for the chance : deal dices to both players
      // actions are saved as dice's face (for example 3) and number of player (for example 1) : d3_1

      d1_1 = 100,
      d2_1,
      d3_1,
      d4_1,
      d5_1,
      dstar_1,
      d1_2 = 200,
      d2_2,
      d3_2,
      d4_2,
      d5_2,
      dstar_2,

    };

    enum class dice_face_t {
      NA = -1,
      face_1 = 1,
      face_2,
      face_3,
      face_4,
      face_5,
      face_star,

      // TODO clean this up
          DEAL1 = -2,
      DEAL2 = -3,
    };

    static constexpr int N_PLAYERS = 2;
    static constexpr int N_DICES = 2;
    static constexpr int MAX_VALUE_DICE = N_DICES * 2 ;
    using action_vector_t = vector<action_t>;


    struct infoset_t : public oz::infoset_t::concept_t {
      const player_t player;
      const dice_face_t face1;  // face of the first dice
      const dice_face_t face2;  // face of the second dice
      const action_vector_t history;
      const array<int, 2> bet;    // n. of dice   dice face
      const int raises_face;
      const int raises_dice;
      const int action_number;  //first one is for the dice face, the second one for the dice numbers


      infoset_t(player_t player, dice_face_t face1, dice_face_t face2,
                action_vector_t history, array<int, 2> bet, int raises_face, int raises_dice, int action_number):
          player(player), face1(face1), face2(face2),
          history(move(history)), bet(bet), raises_face(raises_face), raises_dice(raises_dice), action_number(action_number) { }

      actions_list_t actions() const override;
      string str() const override;
      bool is_equal(const concept_t &that) const override;
      size_t hash() const override;
    };

    void act_(action_t a);

    void act(oz::action_t a) override { act_(a.cast<action_t>()); }
    oz::infoset_t infoset() const override;
    player_t player() const override { return player_; }
    bool is_terminal() const override;
    value_t utility(player_t player) const override;
    oz::infoset_t infoset(oz::infoset_t::allocator_t alloc) const override;
    action_prob_map_t chance_actions() const override;

    std::unique_ptr<game_t> clone() const override {
      return std::make_unique<liar_dice_t>(*this);
    }


    action_prob_map_t chance_actions(action_prob_allocator_t alloc) const override;

    static constexpr int N_ROUNDS = 1;
    static constexpr int MAX_VALUE_FACE = 5;


  private:
    static constexpr action_t CHANCE_START = action_t::d1_1;
    static constexpr action_t CHANCE_FINISH = action_t::dstar_2;

    player_t player_ = CHANCE;
    array<dice_face_t, N_PLAYERS> face1_ { {dice_face_t::NA, dice_face_t::NA} };
    array<dice_face_t, N_PLAYERS> face2_ { {dice_face_t::NA, dice_face_t::NA} };
    array<int, 2> bet_ {{ 0, 0 }};
    int round_ = 0;
    int raises_face_ = 0;
    int raises_dice_ = 0;
    int action_number = 0;
    action_vector_t history_;
    array<bool, N_PLAYERS> called_ {{ false, false }};

    player_t other_player() const {
      assert(player_ == P1 || player_ == P2);
      return player_ == P1 ? P2 : P1;
    }

    inline static int player_idx(player_t p) {
      assert(p == P1 || p == P2);
      switch (p) {
        case P1: return 0;
        case P2: return 1;
        default: return 0; // should not be reachable
      }
    }

    void deal_hand(action_t a);

    static int hand_rank(dice_face_t dice_face);

  public:
    dice_face_t face1(player_t p) const { return face1_[player_idx(p)]; }
    dice_face_t &face1(player_t p) { return face1_[player_idx(p)]; }
    dice_face_t face2(player_t p) const { return face2_[player_idx(p)]; }
    dice_face_t &face2(player_t p) { return face2_[player_idx(p)]; }

    int bet(int n) const { return bet_[n]; }
    int &bet(int n) { return bet_[n]; }

    bool called(player_t p) const { return called_[player_idx(p)]; }
    bool &called(player_t p) { return called_[player_idx(p)]; }

  };

} // namespace oz

#endif // OZ_LD_H
