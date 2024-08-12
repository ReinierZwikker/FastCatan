#include "game_manager.h"


GameManager::GameManager() = default;

GameManager::~GameManager() {
  close_log();
  delete log.game_summaries;
  delete log.moves;
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
  if (log.game_summary_file != nullptr) {
    std::fclose(log.game_summary_file);
  }
  if (log.move_file != nullptr) {
    std::fclose(log.move_file);
  }
}

void GameManager::add_game_to_log() {
  if (log.game_summaries && (log.type == GameLog || log.type == BothLogs)) {
    log.game_summaries->id = games_played;
    log.game_summaries->rounds = game.current_round;
    log.game_summaries->moves_played = log.writes;
    log.game_summaries->run_time = (uint8_t)(run_speed * 1000);  // to ms
    log.game_summaries->winner = game.game_winner;
    log.game_summaries->num_players = game.num_players;
    log.game_summaries->seed = game.seed;
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

void GameManager::add_ai_helper(BeanHelper* bean_ai_helper) {
  bean_helper = bean_ai_helper;
}

void GameManager::add_ai_helper(ZwikHelper* zwik_ai_helper) {
  zwik_helper = zwik_ai_helper;
}

void GameManager::update_ai() {
  if (bean_helper != nullptr) {
    bean_helper->update();
  }
  if (zwik_helper != nullptr) {
    zwik_helper->update();
  }
}

void GameManager::assign_players() {
  Player* players[4];
  uint8_t bean_player_i = 0;
  uint8_t zwik_player_i = 0;
  for (int player_i = 0; player_i < app_info.num_players; ++player_i) {
    players[player_i] = new Player(&game.board, index_color(player_i));
    switch (app_info.selected_players[player_i]) {
      case PlayerType::beanPlayer:
        if (bean_helper != nullptr) {
          players[player_i] = bean_helper->ai_players[bean_player_i];
          players[player_i]->activated = true;
          ++bean_player_i;
        }
        else {
          error_message = {WindowName::AI, (uint8_t)player_i, "Bean Helper was not initialized"};
          break;
        }
        break;
      case PlayerType::zwikPlayer:
        if (zwik_helper != nullptr) {
          players[player_i] = zwik_helper->ai_players[zwik_player_i];
          players[player_i]->activated = true;
          ++zwik_player_i;
        }
        else {
          error_message = {WindowName::AI, (uint8_t)player_i, "Zwik Helper was not initialized"};
          break;
        }
        break;
      case PlayerType::consolePlayer:
        error_message = {WindowName::AI, (uint8_t)player_i, "Console player is not implemented"};
        break;
      case PlayerType::guiPlayer:
        players[player_i]->agent = new GuiPlayer(players[player_i]);
        players[player_i]->activated = true;
        break;
      case PlayerType::randomPlayer:
        players[player_i]->agent = new RandomPlayer(players[player_i]);
        players[player_i]->activated = true;
        break;
      case PlayerType::NoPlayer:
        break;
    }
  }
  game.add_players(players);
}

void GameManager::run_multiple_games() {

  game.log = &log;

  while(keep_running) {
    clock_t begin_clock = clock();

    assign_players();

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

    if (log.type != NoLog) {
      if (seed == 0) {
        game.reseed(rd());
      }
      else {
        game.reseed(seed);
      }
    }

    update_ai();

    game.reset();
    ++games_played;
    clock_t end_clock = clock();
    run_speed = (double)(end_clock - begin_clock) / CLOCKS_PER_SEC;
  }

  write_log_to_disk();
  close_log();
}

