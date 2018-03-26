//
// Created by andy on 26/03/18.
//

#ifndef OZ_FLIPGUESS_H
#define OZ_FLIPGUESS_H

class FlipGuess {
 public:
  enum class Player {
    Chance,
    P1,
    P2
  };

  enum class Action {
    NA,
    Left,
    Right,
    Heads,
    Tails
  };

  class Infoset {

  };

  FlipGuess();
  bool is_terminal();
  int utility();
  Infoset infoset();
  void act();

 private:
  bool finished_;
  bool heads_;
  Player player_;
  Action p1_action_;
  Action p2_action_;
};

#endif // OZ_FLIPGUESS_H
