//
// Created by andy on 26/03/18.
//

#include "FlipGuess.h"

FlipGuess::FlipGuess():
  finished_ { false },
  heads_ { false },
  player_ { Player::Chance },
  p1_action_ { Action::NA },
  p2_action_ { Action::NA }
{}
