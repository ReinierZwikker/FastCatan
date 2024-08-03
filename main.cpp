#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mutex>

#include "app/app.h"
#include "src/game/game.h"

int main(int argc, char *argv[]) {

  std::mutex human_turn;

  int amount_of_players = atoi(argv[1]);

  Game game = Game(amount_of_players);
  App app = App(0, nullptr, &game);

  while(!app.done) {
    app.Refresh();
  }
  return 0;
}

// TODO GuiPlayer
  // TODO Handle RobberTurn

// TODO Robber
  // TODO Show Robber
  // TODO Pull card when moved

// TODO VictoryPoints
  // TODO Longest Road
  // TODO Largest Army

// TODO Trading

// TODO Exchanging with harbor

// TODO Streets Highlight

// TODO Graphics
