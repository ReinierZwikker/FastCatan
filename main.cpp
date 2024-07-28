#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "app/app.h"
#include "src/game/game.h"

int main(int argc, char *argv[]) {

  int amount_of_players = atoi(argv[1]);

  Game game = Game(amount_of_players);

//  game.start_game();

  App app = App(0, nullptr, &game);

  while(!app.done) {
    app.Refresh();
  }
  return 0;
};
