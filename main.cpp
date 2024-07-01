#include <iostream>

#include "game.h"

int main() {

  Game game = Game(3);

  game.start_round();

  game.step_round();
  game.step_round();

  return 0;
}
