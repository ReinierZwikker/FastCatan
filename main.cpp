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
