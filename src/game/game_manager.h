#ifndef FASTCATAN_GAME_MANAGER_H
#define FASTCATAN_GAME_MANAGER_H

#include "game/game.h"
#include "game/components.h"

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

    void run_multiple_games();

    void start_log(LogType log_type, const std::string& filename, const std::filesystem::path& dirPath);
    void close_log() const;
    void add_game_to_log();
    void write_log_to_disk() const;

    Game game = Game(true);
    Logger log{};

  private:

};

#endif //FASTCATAN_GAME_MANAGER_H
