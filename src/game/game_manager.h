#ifndef FASTCATAN_GAME_MANAGER_H
#define FASTCATAN_GAME_MANAGER_H

#include "game/game.h"
#include "game/components.h"
#include "game/AIHelpers/bean_helper.h"
#include "game/AIHelpers/zwik_helper.h"
#include "app_components.h"

#include <atomic>
#include <filesystem>
#include <iostream>

class GameManager {
public:
  GameManager();
  ~GameManager();

  std::atomic<int> id = 0;
  std::atomic<double> run_speed = 0;

  std::atomic<int> total_games_played = 0;
  int games_played = 0;
  std::atomic<bool> keep_running = false;
  std::atomic<bool> finished = false;

  std::atomic<bool> updating = false;
  std::atomic<bool> ready_for_update = true;

  void run_multiple_games();
  void run_single_game();

  void add_seed(unsigned int seed);

  void start_log(LogType log_type, const std::string& filename, const std::filesystem::path& dirPath);
  void close_log() const;
  void add_game_to_log();
  void write_log_to_disk() const;

  Game* game = nullptr;
  Logger log{};

  cudaStream_t cuda_stream;

  AppInfo app_info;
  ErrorMessage error_message;
  void assign_players();
  void add_ai_helper(BeanHelper*);
  void add_ai_helper(ZwikHelper*);
  BeanHelper* bean_helper = nullptr;
  bool bean_helper_active = false;
  ZwikHelper* zwik_helper = nullptr;
  bool zwik_helper_active = false;

private:

  void update_ai();
  void run();

  std::mutex manager_mutex;
  unsigned int seed = 42;
  std::mt19937 gen;
  std::uniform_int_distribution<unsigned int> new_seed;

};

#endif //FASTCATAN_GAME_MANAGER_H
