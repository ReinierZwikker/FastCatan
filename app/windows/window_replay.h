#ifndef FASTCATAN_WINDOW_REPLAY_H
#define FASTCATAN_WINDOW_REPLAY_H

#include "imgui.h"
#include "viewport.h"
#include "game/game_manager.h"
#include "game/components.h"
#include "app_components.h"

#include <thread>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

class WindowReplay {
public:
  WindowReplay();
  ~WindowReplay();

  void show(Game*, ViewPort*, AppInfo*);

private:

  void load_games(const std::string& folder);
  void transfer(const std::string& folder, int game_id);

  GameManager* game_manager;
  GameStates game_state;
  PlayerState player_state;
  std::mutex mutex;

  const int processor_count = (int)std::thread::hardware_concurrency();

  bool play = false;
  unsigned int play_tick = 0;
  float play_speed = 1;

  bool invalid_input_folder = false;
  bool failed_to_load_game = false;
  int current_game = 0;
  int current_move = 1;
  int thread_id = 1;
  std::vector<GameSummary> loaded_games;
  std::vector<Move> loaded_moves;

  // Widgets
  void reset_replay_state(Game* game, AppInfo* app_info);
  void input_folder_box();
  void thread_box();
  void load_button();
  void replay_button(Game* game, AppInfo* app_info);
  void next_move_button(Game* game, AppInfo* app_info);
};

#endif //FASTCATAN_WINDOW_REPLAY_H
