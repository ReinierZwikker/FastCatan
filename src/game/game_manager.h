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

  std::atomic<int> games_played = 0;
  std::atomic<bool> keep_running = false;

  std::mutex manager_mutex;

  std::random_device rd;
  unsigned int seed = 42;

  void run_multiple_games();

  void start_log(LogType log_type, const std::string& filename, const std::filesystem::path& dirPath);
  void close_log() const;
  void add_game_to_log();
  void write_log_to_disk() const;

  Game game = Game(true);
  Logger log{};

  AppInfo app_info;
  ErrorMessage error_message;
  void assign_players();
  void add_ai_helper(BeanHelper*);
  void add_ai_helper(ZwikHelper*);
  BeanHelper* bean_helper = nullptr;
  ZwikHelper* zwik_helper = nullptr;

private:

  void update_ai();

};

#endif //FASTCATAN_GAME_MANAGER_H
