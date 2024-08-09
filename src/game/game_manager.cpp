#include "game_manager.h"


GameManager::GameManager() {

}

GameManager::~GameManager() {
  close_log();
}

void GameManager::start_log(LogType log_type, const std::string& filename,
                            const std::filesystem::path& dirPath = "logs") {
  if (log.file == nullptr) {
    if (!std::filesystem::exists(dirPath)) {
      std::filesystem::create_directory(dirPath);
    }

    log = Logger();
    log.type = log_type;
    log.file = std::fopen(filename.c_str(), "wb");

  }
}

void GameManager::close_log() const {
  if (log.file) {
    std::fclose(log.file);
  }
}

void GameManager::add_game_to_log() {
  ++log.writes;
  if (log.file && log.type == GameLog) {
    log.data += "\nGame: " + std::to_string(games_played) + "\n";
    log.data += "\tRounds:  " + std::to_string(game.current_round) + "\n";
    log.data += "\tWinner:  " + color_names[game.game_winner] + "\n";
    log.data += "\tRunTime: " + std::to_string((int)(run_speed * 1000)) + " ms\n";
  }

  if (log.writes > 500) {
    write_log_to_disk();
    log.writes = 0;
    log.data = "";
  }
}

void GameManager::write_log_to_disk() const {
  if (log.file) {
    const char* data = log.data.c_str();
    std::fprintf(log.file, "%s", data);
  }
}

void GameManager::run_multiple_games() {
  start_log(GameLog, "logs/GameLog_Thread_" + std::to_string(id));

  while(keep_running) {
    clock_t begin_clock = clock();

    game.run_game();
    if (log.type == GameLog) {
      add_game_to_log();
    }
    game.reset();
    ++games_played;

    clock_t end_clock = clock();
    run_speed = (double)(end_clock - begin_clock) / CLOCKS_PER_SEC;
  }

  write_log_to_disk();
  close_log();
}

