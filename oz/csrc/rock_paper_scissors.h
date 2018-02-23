#include <cassert>
#include <string>
#include <cmath>

struct RockPaperScissors {
  enum class Player {
    Chance = -1,
    P1 = 1,
    P2 = 2
  };

  enum class Action {
    NA = -1,
    Rock = 1,
    Paper,
    Scissors
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
  Action action_[N_PLAYERS];
  Player player;

  RockPaperScissors():
      finished { false },
      action_ { Action::NA, Action::NA },
      player { Player::P1 }
  { }

  Infoset infoset(Player p) const {
    return Infoset { player };
  }

  Infoset infoset() const {
    return infoset(player);
  }

  bool is_terminal() const {
    return finished;
  }

  void act(Action a) {
    action(player) = a;

    if (player == Player::P1) {
      player = Player::P2;
    }
    else if (player == Player::P2) {
      finished = true;
    }
  }

  std::vector<Action> legal_actions() {
    return std::vector<Action> {
      Action::Rock,
      Action::Paper,
      Action::Scissors
    };
  }

  int reward() {
    Action a1 = action(Player::P1),
           a2 = action(Player::P2);

    if (a1 == a2) {
      return 0;
    }

    else if (a1 == Action::Rock) {
      if (a2 == Action::Paper) {
        return -1;
      }
      else if (a2 == Action::Scissors) {
        return 1;
      }
    }

    else if (a1 == Action::Paper) {
      if (a2 == Action::Rock) {
        return 1;
      }
      else if (a2 == Action::Scissors) {
        return -1;
      }
    }

    else if (a1 == Action::Scissors) {
      if (a2 == Action::Paper) {
        return 1;
      }
      else if (a2 == Action::Rock) {
        return -1;
      }
    }

    assert (false);
    return 0;
  }

  static int player_idx(Player p) {
    switch (p) {
      case Player::P1: return 0;
      case Player::P2: return 1;
      default:
        throw std::invalid_argument("invalid player");
    }
  }

  Action  action(Player p) const { return action_[player_idx(p)]; }
  Action& action(Player p) { return action_[player_idx(p)]; }
};

inline bool operator==(const RockPaperScissors::Infoset& lhs, const RockPaperScissors::Infoset& rhs) {
  return lhs.player == rhs.player;
}
