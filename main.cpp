#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "game.h"

int main(int argc, char *argv[]) {

  int amount_of_players = atoi(argv[1]);

  printf("\n  ===  FastCatan  ===  \n\nStarting game with %d players!\n", amount_of_players);

  Game game = Game(amount_of_players);

  game.board.PrintBoard();

//  game.start_round();

//  game.step_round();
//  game.step_round();

  return 0;
}
