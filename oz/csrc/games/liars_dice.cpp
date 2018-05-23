#include "liars_dice.h"
#include "hash.h"

#include <cassert>
#include <memory>
#include <vector>
#include <sstream>
#include <iostream>

namespace oz {

  using namespace std;

  void liar_dice_t::act_(action_t a) {

    if (player_ == CHANCE) {
      deal_hand(a);

    }
    else {

      history_.push_back(a);
      if (a == action_t::Call_liar) {
        called(player()) = true;
      }
      else if (a == action_t::Raise_0face) {
        action_number = -1;
      }
      else if (a == action_t::Raise_1face) {


        if (raises_face_ > MAX_VALUE_FACE or (bet(1) + 1) > MAX_VALUE_FACE) {
          throw std::invalid_argument("maximum face raises reached");
        }
        bet(1) += 1;
        raises_face_ += 1;
        action_number = 1;
      }
      else if (a == action_t::Raise_2face) {

        if (raises_face_ > MAX_VALUE_FACE or (bet(1) + 2) > MAX_VALUE_FACE) {
          throw std::invalid_argument("maximum face raises reached");
        }
        bet(1) += 2 ;
        raises_face_ += 1;
        action_number = 1;
      }
      else if (a == action_t::Raise_3face ) {

        if (raises_face_ > MAX_VALUE_FACE or (bet(1) + 3) > MAX_VALUE_FACE) {
          throw std::invalid_argument("maximum face raises reached");
        }
        bet(1) += 3 ;
        raises_face_ += 1;
        action_number = 1;
      }
      else if (a == action_t::Raise_4face ) {

        if (raises_face_ > MAX_VALUE_FACE or (bet(1) + 4) > MAX_VALUE_FACE) {
          throw std::invalid_argument("maximum face raises reached");
        }
        bet(1) += 4 ;
        raises_face_ += 1;
        action_number = 1;
      }
      else if (a == action_t::Raise_5face) {
        if (raises_face_ > MAX_VALUE_FACE or (bet(1) + 5) > MAX_VALUE_FACE) {
          throw std::invalid_argument("maximum face raises reached");
        }
        bet(1) += 5 ;
        raises_face_ += 1;
        action_number = 1;
      }
      else if (a == action_t::Raise_0dice) {

        if (raises_dice_ > MAX_VALUE_DICE) {
          throw std::invalid_argument("maximum dice raises reached");
        }

        player_ = other_player();
        action_number = 0;
      }
      else if (a == action_t::Raise_1dice) {


        if (raises_dice_ >= MAX_VALUE_DICE) {
          throw std::invalid_argument("maximum dice raises reached");
        }

        bet(0) += 1 ;

        raises_dice_ += 1;
        player_ = other_player();
        action_number = 0;

      }
      else if (a == action_t::Raise_2dice) {


        if (raises_dice_ >= MAX_VALUE_DICE) {
          throw std::invalid_argument("maximum dice raises reached");
        }

        bet(0) += 2 ;
        raises_dice_ += 1;

        player_ = other_player();

        action_number = 0;

      }
      else if (a == action_t::Raise_3dice) {


        if (raises_dice_ >= MAX_VALUE_DICE) {
          throw std::invalid_argument("maximum dice raises reached");
        }

        bet(0) += 3;
        raises_dice_ += 1;
        player_ = other_player();
        action_number = 0;

      }
      else if (a == action_t::Raise_4dice) {


        if (raises_dice_ >= MAX_VALUE_DICE) {
          throw std::invalid_argument("maximum dice raises reached");
        }

        bet(0) += 4;
        raises_dice_ += 1;
        player_ = other_player();
        action_number = 0;

      }

      else {
        throw std::invalid_argument("invalid action");
      }
    }
  }



  void liar_dice_t::deal_hand(action_t a) {


    if (!(a >= CHANCE_START && a <= CHANCE_FINISH)) {
      throw std::invalid_argument("illegal action");
    }

    switch (a) {
      case action_t::d1_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_1;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_2;
        break;
      case action_t::d2_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_2;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_2;
        break;
      case action_t::d3_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_3;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_3;
        break;
      case action_t::d4_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_4;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_4;
        break;
      case action_t::d5_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_5;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_5;
        break;
      case action_t::dstar_1:
        if (face1(P1) == dice_face_t::NA)
          face1(P1) = dice_face_t::face_star;
        else if (N_DICES == 2)
          face2(P1) = dice_face_t::face_star;
        break;
      case action_t::d1_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_1;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_1;
        break;
      case action_t::d2_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_2;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_2;
        break;
      case action_t::d3_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_3;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_3;
        break;
      case action_t::d4_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_4;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_4;
        break;
      case action_t::d5_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_5;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_5;
        break;
      case action_t::dstar_2:
        if (face1(P2) == dice_face_t::NA)
          face1(P2) = dice_face_t::face_star;
        else if (N_DICES == 2)
          face2(P2) = dice_face_t::face_star;
        break;


      default: assert(false);
    }

    switch (a) {
      case action_t::d1_1:
      case action_t::d2_1:
      case action_t::d3_1:
      case action_t::d4_1:
      case action_t::d5_1:
      case action_t::dstar_1:
        player_ = CHANCE;
        break;

      case action_t::d1_2:
      case action_t::d2_2:
      case action_t::d3_2:
      case action_t::d4_2:
      case action_t::d5_2:
      case action_t::dstar_2:
        if(face2(P2) == dice_face_t::NA)
          if(N_DICES == 2)
            player_ = CHANCE;
          else
            player_ = P1;
        else
          player_ = P1;
        break;


      default: assert(false);
    }
  }


  auto liar_dice_t::utility(player_t player) const -> value_t {
    assert (is_terminal());

    value_t u = 0;
    int count_value = 0;
    if(hand_rank(face1(P1)) == bet(1) or hand_rank(face1(P1)) == 6)
      count_value += 1;

    if(hand_rank(face2(P1)) == bet(1) or hand_rank(face2(P1)) == 6)
      count_value += 1;

    if(hand_rank(face1(P2)) == bet(1) or hand_rank(face1(P2)) == 6)
      count_value += 1;

    if(hand_rank(face2(P2)) == bet(1) or hand_rank(face2(P2)) == 6)
      count_value += 1;


    if(called(P1) and count_value == bet(0))
      u = -1;
    else if (called(P2) and count_value == bet(0))
      u = 1;
    else if (called(P1) and count_value != bet(0))
      u = 1;
    else
      u = -1;




    return relative_utility(player, u);
  }

  auto liar_dice_t::hand_rank(dice_face_t dice_face) -> int {


    if (dice_face == dice_face_t::face_1) {
      return 1;
    }
    else if (dice_face == dice_face_t::face_2) {
      return 2;
    }
    else if (dice_face == dice_face_t::face_3) {
      return 3;
    }
    else if (dice_face == dice_face_t::face_4) {
      return 4;
    }
    else if (dice_face == dice_face_t::face_5) {
      return 5;
    }
    else if (dice_face == dice_face_t::face_star) {
      return 6;
    }
    else {
      return  -1;
    }
  }

  auto liar_dice_t::infoset() const -> oz::infoset_t {

    Expects(player() != CHANCE);
    return make_infoset<infoset_t>(player_, face1(player_), face2(player_),
                                   history_, bet_, raises_face_, raises_dice_, action_number);

  }

  auto liar_dice_t::infoset(oz::infoset_t::allocator_t alloc) const
  -> oz::infoset_t
  {
    Expects(player() != CHANCE);
    return allocate_infoset<infoset_t, oz::infoset_t::allocator_t>
        (alloc,
         player_, face1(player_), face2(player_),
         history_, bet_, raises_face_, raises_dice_, action_number);
  }

  auto liar_dice_t::is_terminal() const -> bool {
    return called(P1) || called(P2) || round_ >= N_ROUNDS;
  }


  auto liar_dice_t::infoset_t::actions() const -> actions_list_t {


    static const actions_list_t raise_40face_call{
        make_action(action_t::Raise_0face),
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Raise_3face),
        make_action(action_t::Raise_4face),
        make_action(action_t::Call_liar),
    };

    static const actions_list_t raise_4face_call{
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Raise_3face),
        make_action(action_t::Raise_4face),
        make_action(action_t::Call_liar),
    };

    static const actions_list_t raise_3face_call{
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Raise_3face),
        make_action(action_t::Call_liar),
    };
    static const actions_list_t raise_30face_call{
        make_action(action_t::Raise_0face),
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Raise_3face),
        make_action(action_t::Call_liar),
    };
    static const actions_list_t raise_2face_call{
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Call_liar),
    };
    static const actions_list_t raise_20face_call{
        make_action(action_t::Raise_0face),
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Call_liar),
    };

    static const actions_list_t raise_1face_call{
        make_action(action_t::Raise_1face),
        make_action(action_t::Call_liar),
    };
    static const actions_list_t raise_10face_call{
        make_action(action_t::Raise_0face),
        make_action(action_t::Raise_1face),
        make_action(action_t::Call_liar),
    };
    static const actions_list_t raise_0dice{
        make_action(action_t::Raise_0dice),
    };
    static const actions_list_t raise_10dice{
        make_action(action_t::Raise_0dice),
        make_action(action_t::Raise_1dice),
    };
    static const actions_list_t raise_1dice{
        make_action(action_t::Raise_1dice),
    };
    static const actions_list_t raise_20dice{
        make_action(action_t::Raise_0dice),
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
    };
    static const actions_list_t raise_2dice{
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
    };
    static const actions_list_t raise_30dice{
        make_action(action_t::Raise_0dice),
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
        make_action(action_t::Raise_3dice),
    };
    static const actions_list_t raise_3dice{
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
        make_action(action_t::Raise_3dice),
    };
    static const actions_list_t raise_40dice{
        make_action(action_t::Raise_0dice),
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
        make_action(action_t::Raise_3dice),
        make_action(action_t::Raise_4dice),
    };
    static const actions_list_t raise_4dice{
        make_action(action_t::Raise_1dice),
        make_action(action_t::Raise_2dice),
        make_action(action_t::Raise_3dice),
        make_action(action_t::Raise_4dice),
    };


    static const actions_list_t raise_all_face{
        make_action(action_t::Raise_1face),
        make_action(action_t::Raise_2face),
        make_action(action_t::Raise_3face),
        make_action(action_t::Raise_4face),
        make_action(action_t::Raise_5face)

    };

    static const actions_list_t call{
        make_action(action_t::Call_liar),

    };


    if (raises_face == 0 and action_number == 0) {
      return raise_all_face;
    } else if (raises_dice == 0 and action_number == 1) {
      if (N_DICES == 2)
        return raise_4dice;
      else {
        return raise_2dice;
      }
    } else if (raises_face < MAX_VALUE_FACE and bet[1] < 2 and action_number == 0) {
      if(bet[0] == 2 and N_DICES == 1) {
        return raise_4face_call;

      }
      else {

        return raise_40face_call;
      }
    } else if (raises_dice < MAX_VALUE_DICE and bet[0] < 2 and action_number == 1) {
      if (N_DICES == 2)
        return raise_30dice;
      else {

        return raise_10dice;
      }
    } else if (raises_dice < MAX_VALUE_DICE and bet[0] < 2 and action_number == -1) {

      if (N_DICES == 2)
        return raise_3dice;
      else {

        return raise_1dice;

      }
    }
    else if (raises_face < MAX_VALUE_FACE and bet[1] < 3 and action_number == 0) {
      if (bet[0] == 2 and N_DICES == 1) {

        return raise_3face_call;
      }
      else {

        return raise_30face_call;
      }
    }
    else if (bet[1] == 3 and N_DICES == 1 and action_number == 1) {

      return raise_0dice;
    }else if (raises_dice < MAX_VALUE_DICE and bet[0] < 3 and action_number == 1 and N_DICES == 2 ) {
      return raise_20dice;
    } else if (raises_dice < MAX_VALUE_DICE and bet[0] < 3 and action_number == -1) {
      if(N_DICES == 1 and bet[0] == 2) {
        return raise_0dice;
      }
      else
        return raise_2dice;
    } else if (raises_face < MAX_VALUE_FACE and bet[1] < 4 and action_number == 0) {

      return raise_1face_call;
    }

    else if (action_number == 1 and N_DICES == 1 and bet[0] == 2)
      return  raise_0dice;
    else {

      return call;
    }
  }



  auto liar_dice_t::chance_actions(action_prob_allocator_t alloc) const -> action_prob_map_t {

    Expects(player() == CHANCE);

    static const vector<oz::action_t> chance_actions_p1 {
        make_action(action_t::d1_1),
        make_action(action_t::d2_1),
        make_action(action_t::d3_1),
        make_action(action_t::d4_1),
        make_action(action_t::d5_1),
        make_action(action_t::dstar_1),
    };

    static const vector<oz::action_t> chance_actions_p2 {
        make_action(action_t::d1_2),
        make_action(action_t::d2_2),
        make_action(action_t::d3_2),
        make_action(action_t::d4_2),
        make_action(action_t::d5_2),
        make_action(action_t::dstar_2),
    };


    float p = 1 / 6.0;


    if (face1(P1) == dice_face_t::NA) {
      liar_dice_t::action_prob_map_t m(alloc);
      for (auto action_it = begin(chance_actions_p1); action_it != end(chance_actions_p1); ++action_it) {
        Ensures(0 <= p && p <= 1);
        m.emplace(*action_it, p);
      }
      return m;
    }
    else if(N_DICES == 2 and face2(P1) == dice_face_t::NA) {
      liar_dice_t::action_prob_map_t m(alloc);
      for (auto action_it = begin(chance_actions_p1); action_it != end(chance_actions_p1); ++action_it) {
        m.emplace(*action_it, p);
        Ensures(0 <= p && p <= 1);
        m.emplace(*action_it, p);
      }
      return m;
    }
    else if (face1(P2) == dice_face_t::NA) {
      liar_dice_t::action_prob_map_t m(alloc);
      for (auto action_it = begin(chance_actions_p2); action_it != end(chance_actions_p2); ++action_it) {
        Ensures(0 <= p && p <= 1);
        m.emplace(*action_it, p);
      }
      return m;
    }
    else if(N_DICES == 2 and face2(P2) == dice_face_t::NA) {
      liar_dice_t::action_prob_map_t m(alloc);
      for (auto action_it = begin(chance_actions_p2); action_it != end(chance_actions_p2); ++action_it) {
        m.emplace(*action_it, p);
        Ensures(0 <= p && p <= 1);
        m.emplace(*action_it, p);
      }
      return m;
    }

    assert (false);
    return { };

  }

  auto liar_dice_t::chance_actions() const -> action_prob_map_t {
    return liar_dice_t::chance_actions({ });
  }

  auto liar_dice_t::infoset_t::str() const -> std::string {

    stringstream ss;

    if (face1 == dice_face_t::face_1) {
      ss << "1";
    }
    else if (face1 == dice_face_t::face_2) {
      ss << "2";
    }
    else if (face1 == dice_face_t::face_3) {
      ss << "3";
    }
    else if (face1 == dice_face_t::face_4) {
      ss << "4";
    }
    else if (face1 == dice_face_t::face_5) {
      ss << "5";
    }
    else if (face1 == dice_face_t::face_star) {
      ss << "star";
    }

    if (face2 == dice_face_t::face_1) {
      ss << "1";
    }
    else if (face2 == dice_face_t::face_2) {
      ss << "2";
    }
    else if (face2 == dice_face_t::face_3) {
      ss << "3";
    }
    else if (face2 == dice_face_t::face_4) {
      ss << "4";
    }
    else if (face2 == dice_face_t::face_5) {
      ss << "5";
    }
    else if (face2 == dice_face_t::face_star) {
      ss << "star";
    }


    if (!history.empty()) {
      ss << "/";
    }

    for (const auto& a : history) {
      if (a == action_t::Raise_0face) {
        ss << "No raise face ";
      }
      else if (a == action_t::Raise_1face) {
        ss << "raise_1face ";
      }
      else if (a == action_t::Raise_2face) {
        ss << "raise_2face ";
      }
      else if (a == action_t::Raise_3face) {
        ss << "raise_3face ";
      }
      else if (a == action_t::Raise_4face) {
        ss << "raise_4face ";
      }
      else if (a == action_t::Raise_5face) {
        ss << "raise_5face ";
      }
      else if (a == action_t::Raise_0dice) {
        ss << "No raise dice ";
      }
      else if (a == action_t::Raise_1dice) {
        ss << "raise_1dice ";
      }
      else if (a == action_t::Raise_2dice) {
        ss << "raise_2dice ";
      }
      else if (a == action_t::Raise_3dice) {
        ss << "raise_3dice ";
      }
      else if (a == action_t::Raise_4dice) {
        ss << "riase_4dice ";
      }
      else if (a == action_t::Call_liar) {
        ss << "c ";
      }
      else if (a == action_t::NextRound) {
        ss << "/ ";
      }
      else { assert (false); }
    }

    return ss.str();
  }


  bool liar_dice_t::infoset_t::is_equal(const infoset_t::concept_t &that) const {

    if (typeid(*this) == typeid(that)) {
      auto that_ = static_cast<const liar_dice_t::infoset_t &>(that);
      return player == that_.player &&
             face1 == that_.face1 &&
             face2 == that_.face2 &&
             history == that_.history &&
             bet == that_.bet &&
             raises_face == that_.raises_face;
    }
    else {
      return false;
    }
  }

  size_t liar_dice_t::infoset_t::hash() const {
    size_t seed = 0;
    hash_combine(seed, player);
    hash_combine(seed, face1);
    hash_combine(seed, face2);
    for (const auto &a : history) {
      hash_combine(seed, a);
    }
    hash_combine(seed, bet[0]);
    hash_combine(seed, bet[1]);
    hash_combine(seed, raises_face);
    hash_combine(seed, raises_dice);
    return seed;
  }

} // namespace oz
