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

  FlipGuess():
    finished { false },
    p1_action { Action::NA },
    player { Player::Chance },
    heads_ { true }
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
      if (a == Action::Left) {
        if (heads_) {
          finished = true;
        }
        else {
          p1_action = Action::Left;
        }
      }
    }
    else if (player == Player::P2) {
      if (a == p1_action) {
        finished = true;
      }
      else {

      }
    }
  }
};
