#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include "player.h"
#include "board.h"

struct Move {
  // Move template, only set applicable fields when communicating moves
  MoveType move_type = NoMove;
  int index = -1;
  Corner *corner = nullptr;
  Corner *street = nullptr;
  Color other_player = NoColor;
  CardType rx_card = NoCard;
  CardType tx_card = NoCard;
  int amount = -1;
};

struct Game {

  explicit Game(int num_players);

  // TODO generate random game seed
  int game_seed = 1;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player players[4] = {Player(nullptr, NoColor),
                       Player(nullptr, NoColor),
                       Player(nullptr, NoColor),
                       Player(nullptr, NoColor)};

  Board board = Board();

  int current_round = 0;

  void start_game();

  void step_round();

  int roll_dice();

  void give_cards(int rolled_number);

  bool CheckValidity();
  bool CheckValidity(Move move);
  bool CheckValidity(Move move, MoveType move_type);
};

#endif //FASTCATAN_GAME_H
