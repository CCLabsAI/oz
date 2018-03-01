#include <string>
#include <vector>

struct FlipGuess {
  enum class Player {
    Chance = -1,
    P1 = 1,
    P2 = 2
  };

  enum class Action {
    NA = -1,
    Left = 1,
    Right,

    Heads = 100,
    Tails
  };

  struct Infoset {
    Player player;

    std::string str() const {
      if (player == Player::P1) {
        return "P1";
      }
      else if (player == Player::P2) {
        return "P2";
      }
      else {
        assert (false);
        return "";
      }
    }
  };

  static constexpr int N_PLAYERS = 2;
  static constexpr int N_ACTIONS = 3;

  bool finished;
  Action p1_action;
  Player player;
  bool heads_;
  int reward_;

  FlipGuess():
    finished { false },
    p1_action { Action::NA },
    player { Player::Chance },
    heads_ { true },
    reward_ { 0 }
    {}

  bool is_terminal() const {
    return finished;
  }

  void act(Action a) {
    if (player == Player::Chance) {
      if (a == Action::Heads) {
        heads_ = true;
      }
      else if (a == Action::Tails) {
        heads_ = false;
      }

      player = Player::P1;
    }

    else if (player == Player::P1) {
      p1_action = a;

      if (heads_) {
        finished = true;

        if (a == Action::Left) {
          reward_ = 1;
        }
        else {
          reward_ = 0;
        }
      }

      player = Player::P2;
    }

    else if (player == Player::P2) {
      if (a == p1_action) {
        reward_ = 3;
      }
      else {
        reward_ = 0;
      }

      finished = true;
    }
  }

  Infoset infoset(Player p) const {
    return Infoset { player };
  }

  Infoset infoset() const {
    return infoset(player);
  }

  int reward() const {
    return reward_;
  }

  std::vector<Action> legal_actions() {
    if (player == Player::Chance) {
      return std::vector<Action> { Action::Heads, Action::Tails };
    }

    else if (player == Player::P1 || player == Player::P2) {
      return std::vector<Action> { Action::Left, Action::Right };
    }

    else {
      assert(false);
      return std::vector<Action> {};
    }
  }
};

inline bool operator==(const FlipGuess::Infoset& lhs, const FlipGuess::Infoset& rhs) {
  return lhs.player == rhs.player;
}
