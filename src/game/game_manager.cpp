#include "game_manager.h"


GameManager::GameManager() {

}

GameManager::~GameManager() {
  close_log();
}

void GameManager::start_log(LogType log_type, const std::string& filename,
                            const std::filesystem::path& dirPath = "logs") {
  if (log.game_summaries == nullptr && log.move_file == nullptr) {
    if (!std::filesystem::exists(dirPath)) {
      std::filesystem::create_directory(dirPath);
    }

    log = Logger();
    if (log_type == GameLog || log_type == BothLogs) {
      // Assign heap memory to store all possible moves in a game
      log.game_summaries = new GameSummary[10];

      std::string game_summary_filename = filename + "_game_summaries.dat";
      log.game_summary_file = std::fopen(game_summary_filename.c_str(), "wb");
    }
    if (log_type == MoveLog || log_type == BothLogs) {
      // Assign heap memory to store all possible moves in a game
      log.moves = new Move[(moves_per_turn + 5) * max_rounds];

      std::string move_filename = filename + "_moves.dat";
      log.move_file = std::fopen(move_filename.c_str(), "wb");
    }
    log.type = log_type;
  }
}

void GameManager::close_log() const {
  if (log.game_summary_file) {
    std::fclose(log.game_summary_file);
    delete log.game_summaries;
  }
  if (log.move_file) {
    std::fclose(log.move_file);
    delete log.moves;
  }
}

void GameManager::add_game_to_log() {
  if (log.game_summaries && (log.type == GameLog || log.type == BothLogs)) {
    log.game_summaries->id = games_played;
    log.game_summaries->rounds = game.current_round;
    log.game_summaries->moves_played = log.writes;
    log.game_summaries->run_time = (uint8_t)(run_speed * 1000);
    log.game_summaries->winner = game.game_winner;
  }
}

void GameManager::write_log_to_disk() const {
  if (log.game_summaries) {
    std::fwrite(log.game_summaries, sizeof(GameSummary), 1, log.game_summary_file);
  }
  if (log.move_file) {
    // TODO : actually use the buffer
    for (int i = 0; i < log.writes; ++i) {
      std::fwrite(&log.moves[i], sizeof(Move), 1, log.move_file);
    }
  }
}

void GameManager::run_multiple_games() {

  game.log = &log;

  while(keep_running) {
    clock_t begin_clock = clock();

    if (log.type == MoveLog || log.type == BothLogs) {
      Move move;
      move.type = MoveType::Replay;
      move.index = games_played;
      log.moves[0] = move;
      ++log.writes;
    }

    game.run_game();

    if (log.type == GameLog || log.type == BothLogs) {
      add_game_to_log();
    }

    write_log_to_disk();
    log.writes = 0;

    game.reset();
    ++games_played;

    clock_t end_clock = clock();
    run_speed = (double)(end_clock - begin_clock) / CLOCKS_PER_SEC;
  }

  write_log_to_disk();
  close_log();
}

