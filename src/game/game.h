#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include <random>

#include "player.h"
#include "board.h"
//#include "HumanPlayer/console_player.h"
#include "HumanPlayer/gui_player.h"
//#include "AIPlayer/ai_zwik_player.h"
#include "AIPlayer/random_player.h"

#include <mutex>
#include <condition_variable>

enum GameStates {
  UnInitialized,
  ReadyToStart,
  SetupRound,
  SetupRoundFinished,
  PlayingRound,
  RoundFinished,
  GameFinished,
  WaitingForPlayer,
};

static const char* game_states[] = {
    "UnInitialized",
    "Ready to start",
    "Setup Round",
    "Setup Round Finished",
    "Playing Round",
    "Round Finished",
    "Game Finished",
    "Waiting for player"
};

struct Game {
  explicit Game(int num_players);
  ~Game();

  // Handle game state
  GameStates game_state = UnInitialized;
  Color game_winner = NoColor;
  std::mutex human_turn;
  std::condition_variable cv;
  Move gui_moves[4]{};
  void human_input_received();

  std::random_device randomDevice;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player *players[4]{};
  Player *current_player;
  int current_player_id = 0;

  Board board = Board();

  int current_round = 0;

  void start_game();

  void step_round();



  int roll_dice();
  int die_1 = 0;
  int die_2 = 0;

  void give_cards(int rolled_number);

};

#endif //FASTCATAN_GAME_H
