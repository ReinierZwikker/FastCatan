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
#include <atomic>

struct Game {
  explicit Game(int num_players = 4);
  ~Game();
  void add_players();

  // Handle game state
  std::atomic<bool> keep_running = true;
  std::atomic<int> games_played = 0;
  GameStates game_state = UnInitialized;
  Color game_winner = NoColor;

  bool gui_controlled = false;

  // Threading
  bool move_lock_opened;
  std::mutex move_lock;
  std::condition_variable cv;
  Move gui_moves[4]{};
  std::mutex mutex;
  void human_input_received();

  std::random_device randomDevice;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player *players[4]{};
  Player *current_player;
  int current_player_id = 0;

  Board board = Board();

  int current_round = 0;
  Move chosen_move;

  void unavailable_move(Move move, std::string);

  void start_game();
  void step_round();
  void run_game();
  void run_multiple_games();
  void reset();

  int roll_dice();
  int die_1 = 0;
  int die_2 = 0;

  void give_cards(int rolled_number);

};

#endif //FASTCATAN_GAME_H
