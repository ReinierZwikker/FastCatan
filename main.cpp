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

//  for (int i = 0; i < 50000; i++) {
//    if (game.game_state == ReadyToStart) {
//      game.run_game();
//      std::cout << i << ", turns: " << game.current_round << std::endl;
//      game.reset();
//    }
//    else {
//      std::cout << "State: " << game_states[game.game_state] << std::endl;
//    }
//  }

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
