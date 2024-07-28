#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include "player.h"
#include "board.h"
#include "HumanPlayer/human_player.h"

struct Game {

  explicit Game(int num_players);
  ~Game();

  // TODO generate random game seed
  int game_seed = 1;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player *players[4]{};

  Board board = Board();

  int current_round = 0;

  void start_game();

  void step_round();



  int roll_dice();

  void give_cards(int rolled_number);

};

#endif //FASTCATAN_GAME_H
