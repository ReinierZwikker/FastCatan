#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include "player.h"
#include "board.h"
#include "HumanPlayer/human_player.h"

#include <mutex>
#include <condition_variable>

enum GameStates {
  UnInitialized,
  ReadyToStart,
  Starting,
  WaitingForPlayer,
};

static const char* game_states[] = {
    "UnInitialized",
    "Ready to start",
    "Starting",
    "Waiting for player"
};

struct Game {
  explicit Game(int num_players);
  ~Game();

  // Handle game state
  GameStates game_state = UnInitialized;
  std::mutex human_turn;
  std::condition_variable cv;
  bool input_received = false;
  void human_input_received();

  // TODO generate random game seed
  int game_seed = 1;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player *players[4]{};
  Player *current_player;

  Board board = Board();

  int current_round = 0;

  void start_game();

  void step_round();



  int roll_dice();

  void give_cards(int rolled_number);

};

#endif //FASTCATAN_GAME_H
