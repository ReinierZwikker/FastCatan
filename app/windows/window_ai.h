#ifndef FASTCATAN_WINDOW_AI_H
#define FASTCATAN_WINDOW_AI_H

#include "imgui.h"
#include "viewport.h"
#include "game/game_manager.h"
#include "game/components.h"
#include "app_components.h"

#include <thread>
#include <iostream>
#include <mutex>


class WindowAI {
public:
  WindowAI();
  ~WindowAI();

  void show(Game* game, AppInfo* app_info);
  void stop_threads();

private:
  unsigned int seed = 0;

  const unsigned int processor_count = std::thread::hardware_concurrency();
  int num_threads = 30;

  int set_epoch_length = 0;
  bool continue_after_epoch = true;

  int games_played[30];
  unsigned int total_games_played = 0;

  // TODO : Make size depend on processor_count
  GameManager game_managers[30];
  std::thread threads[30];
  std::mutex mutex;

  // Logging
  int log_type = 0;

  bool closing_training = false;

  // AI Helpers
  BeanHelper* bean_helper = nullptr;
  bool bean_helper_active = false;
  int bean_pop_size = 50;
  int bean_seed = 42;
  bool bean_cuda = false;
  bool randomize_seed = false;
  bool log_bean_games = true;
  int bean_updates = 0;
  int bean_evolutions = 0;
  int bean_shuffle_rate = 200;
  int bean_epoch = 1000;

  ZwikHelper* zwik_helper = nullptr;
  bool zwik_helper_active = false;
  int zwik_pop_size = 200;

  // Show bools
  bool show_select_players_menu = false;
  bool show_bean_ai_menu = false;
  bool show_zwik_ai_menu = false;
  uint8_t show_player_error[4] = {0, 0, 0, 0};

  // Widgets
  AppInfo* app_info = nullptr;
  void train_button();
  void stop_training_button();
  void select_players_window();
  void bean_ai_window(Game* game);
  void zwik_ai_window(Game* game);
  void thread_table(Game* game);
};


#endif //FASTCATAN_WINDOW_AI_H
