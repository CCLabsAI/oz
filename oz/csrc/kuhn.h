#include <cassert>
#include <stdexcept>
#include <vector>
#include <sstream>


struct KuhnPoker {
  enum class Action {
    // Player Actions
    Pass = 1,
    Bet,

    // Chance Actions
    JQ = 100,
    JK,
    QJ,
    QK,
    KJ,
    KQ
  };

  enum class Card {
    NA = -1,
    Jack = 1,
    Queen,
    King,
  };

  enum class Player {
    Chance = -1,
    P1 = 0,
    P2 = 1
  };

  struct Infoset {
    const Card hand;
    const std::vector<Action> history;

    std::string str() const {
      std::stringstream ss;

      if (hand == Card::Jack) {
        ss << "J";
      }
      else if (hand == Card::Queen) {
        ss << "Q";
      }
      else if (hand == Card::King) {
        ss << "K";
      }

      if (!history.empty()) {
        ss << "/";
      }

      for (const auto& a : history) {
        if (a == Action::Bet) {
          ss << "b";
        }
        else if (a == Action::Pass) {
          ss << "p";
        }
        else { assert (false); }
      }

      return ss.str();
    }
  };

  static constexpr int N_PLAYERS = 2;
  static constexpr int ANTE = 1;
  static constexpr Action CHANCE_START  = Action::JQ;
  static constexpr Action CHANCE_FINISH = Action::KQ;

  bool showdown;
  bool folded_[N_PLAYERS];
  Card hand_[N_PLAYERS];
  int pot_[N_PLAYERS];
  Player player;

  std::vector<Action> history;

  KuhnPoker():
      showdown { false },
      folded_ { false, false },
      hand_ { Card::NA, Card::NA },
      pot_ { ANTE, ANTE },
      player { Player::Chance }
  { }

  bool is_terminal() const {
    return showdown || 
      folded(Player::P1) || 
      folded(Player::P2);
  }

  Infoset infoset(Player p) const {
    return Infoset {
      hand(p),
      history
    };
  }

  Infoset infoset() const {
    return infoset(player);
  }

  int reward() const {
    Player winner;

    if (showdown) {
      winner = hand(Player::P1) > hand(Player::P2) ?
               Player::P1 :
               Player::P2;
    }
    else if (folded(Player::P1)) {
      winner = Player::P2;
    }
    else if (folded(Player::P2)) {
      winner = Player::P1;
    }
    else {
      assert (false);
      return 0;
    }

    if (winner == Player::P1) {
      return pot(Player::P2);
    }
    else if (winner == Player::P2) {
      return -pot(Player::P1);
    }
    else {
      assert (false);
      return 0;
    }
  }

  std::vector<Action> legal_actions() {
    if (player == Player::Chance) {
      auto&& v = std::vector<Action>();

      int first = static_cast<int>(CHANCE_START),
          last  = static_cast<int>(CHANCE_FINISH);

      for (int a = first; a <= last; a++) {
        v.push_back(static_cast<Action>(a));
      }

      return v;
    } else {
      return std::vector<Action> { Action::Pass, Action::Bet };
    }
  }

  void act(Action a) {
    if (player != Player::Chance) {
      history.push_back(a);
    }

    if (player == Player::Chance) {
      deal_hand(a);
      player = Player::P1;
    }
    else if (player == Player::P1 && a == Action::Pass) {
      if (pot(Player::P2) > ANTE) {
        folded(Player::P1) = true;
      }

      player = Player::P2;
    }
    else if (player == Player::P1 && a == Action::Bet) {
      pot(Player::P1) += 1;
      player = Player::P2;
    }
    else if (player == Player::P2 && a == Action::Pass) {
      if (pot(Player::P1) > ANTE) {
        folded(Player::P2) = true;
      }
      else {
        showdown = true;
      }

      player = Player::P1;
    }
    else if (player == Player::P2 && a == Action::Bet) {
      pot(Player::P2) += 1;
      player = Player::P1;
    }

    if (pot(Player::P1) == 2 && pot(Player::P2) == 2) {
      showdown = true;
    }
  }

  void deal_hand(Action a) {
    if (!(a >= CHANCE_START && a <= CHANCE_FINISH)) {
      throw std::invalid_argument("illegal action");
    }

    switch(a) {
      case Action::JQ:
        hand(Player::P1) = Card::Jack;
        hand(Player::P2) = Card::Queen;
        break;
      case Action::JK:
        hand(Player::P1) = Card::Jack;
        hand(Player::P2) = Card::King;
        break;
      case Action::QJ:
        hand(Player::P1) = Card::Queen;
        hand(Player::P2) = Card::Jack;
        break;
      case Action::QK:
        hand(Player::P1) = Card::Queen;
        hand(Player::P2) = Card::King;
        break;
      case Action::KJ:
        hand(Player::P1) = Card::King;
        hand(Player::P2) = Card::Jack;
        break;
      case Action::KQ:
        hand(Player::P1) = Card::King;
        hand(Player::P2) = Card::Queen;
        break;

      default:
        assert (false);
    }
  }

  static int player_idx(Player p) {
    switch (p) {
      case Player::P1: return 0;
      case Player::P2: return 1;
      default:
        throw std::invalid_argument("invalid player");
    }
  }

  Card  hand(Player p) const { return hand_[player_idx(p)]; }
  Card& hand(Player p) { return hand_[player_idx(p)]; }

  int  pot(Player p) const { return pot_[player_idx(p)]; }
  int& pot(Player p) { return pot_[player_idx(p)]; }

  bool  folded(Player p) const { return folded_[player_idx(p)]; }
  bool& folded(Player p) { return folded_[player_idx(p)]; }
};

inline bool operator==(const KuhnPoker::Infoset& lhs, const KuhnPoker::Infoset& rhs) {
  return lhs.hand == rhs.hand &&
         lhs.history == rhs.history;
}
