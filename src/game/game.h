#ifndef FASTCATAN_GAME_H
#define FASTCATAN_GAME_H

#include <random>
#include <algorithm>

#include "player.h"
#include "board.h"
#include "AIPlayer/random_player.h"
#include "HumanPlayer/console_player.h"
#include "HumanPlayer/gui_player.h"
//#include "AIPlayer/ai_zwik_player.h"

#include <mutex>
#include <condition_variable>

struct Game {
  explicit Game(bool gui = false, int num_players = 4, unsigned int input_seed = 42);
  ~Game();

  void add_player(PlayerType player_type, int player_id);
  void add_players(PlayerType player_type[4]);
  void add_players(Player* new_players[4]);

  // Handle game state
  GameStates game_state = UnInitialized;
  Color game_winner = Color::NoColor;

  bool gui_controlled = false;

  // Threading
  bool move_lock_opened = false;
  std::mutex move_lock;
  std::condition_variable cv;
  std::mutex mutex;
  void human_input_received(Move move) const;

  int num_players;
  // Player order: [Green, Red, White, Blue]
  Player *players[4]{};
  Player *current_player = nullptr;
  int current_player_id = 0;

  // Victory items
  Player *longest_road_player;
  Player *most_knights_player;

  Board board = Board();

  int current_round = 0;
  Move chosen_move;

  // Bank
  DevelopmentType development_cards[amount_of_development_cards]{};
  int current_development_card = 0;

  void unavailable_move(Move move, const std::string&);

  void start_game();
  void step_round();
  void run_game();
  void reset();

  int roll_dice();
  int die_1 = 0;
  int die_2 = 0;
  void reseed(unsigned int input_seed);
  unsigned int seed = 42;
  std::mt19937 gen;
  std::uniform_int_distribution<> dice;
  std::uniform_int_distribution<> card;

  void check_longest_road();
  void check_longest_road_interrupt();
  void check_knights_played();
  void move_robber(int tile_index);
  void shuffle_development_cards();
  void give_cards(int rolled_number);

  // Logging
  void add_move_to_log(Move move) const;
  Logger* log = nullptr;
  unsigned int move_id = 0;

};

#endif //FASTCATAN_GAME_H
