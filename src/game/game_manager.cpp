#include "game_manager.h"


GameManager::GameManager() : new_seed(0, std::numeric_limits<unsigned int>::max()), gen(42) {

  game = new Game(true);

}

GameManager::~GameManager() {
  delete game;
  game = nullptr;

  close_log();
  delete log.game_summaries;
  delete log.moves;
  log.game_summaries = nullptr;
  log.moves = nullptr;
}

void GameManager::add_seed(unsigned int input_seed) {
  seed = input_seed;
  gen.seed(input_seed);
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
    log.game_summaries->id = total_games_played;
    log.game_summaries->rounds = game->current_round;
    log.game_summaries->moves_played = log.writes;
    log.game_summaries->run_time = (uint8_t)(run_speed * 1000);  // to ms
    log.game_summaries->winner = game->game_winner;
    log.game_summaries->num_players = game->num_players;
    log.game_summaries->seed = game->seed;

    for (int player_i = 0; player_i < game->num_players; ++player_i) {
      log.game_summaries->seed_players[player_i] = game->players[player_i]->agent->agent_seed;
      log.game_summaries->type_players[player_i] = game->players[player_i]->agent->get_player_type();
    }
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
  bean_helper_active = true;
}

void GameManager::add_ai_helper(ZwikHelper* zwik_ai_helper) {
  zwik_helper = zwik_ai_helper;
  zwik_helper_active = true;
}

void GameManager::update_ai() {
  if (bean_helper_active) {
    bean_helper->update(game, id, games_played);
  }
  if (zwik_helper_active) {
    zwik_helper->update(game);
  }
}

void GameManager::assign_players() {
  Player* players[4];
  uint8_t bean_player_i = 0;
  uint8_t zwik_player_i = 0;
  for (int player_i = 0; player_i < app_info.num_players; ++player_i) {
    switch (app_info.selected_players[player_i]) {
      case PlayerType::beanPlayer:
        if (bean_helper != nullptr) {
          manager_mutex.lock();
          players[player_i] = bean_helper->ai_total_players[id][bean_player_i];
//          if (id == 0) {
//            players[player_i]->agent->add_cuda(&cuda_stream);
//          }
          manager_mutex.unlock();
          players[player_i]->activated = true;
          game->assigned_players[player_i] = true;
          ++bean_player_i;
        }
        else {
          throw std::invalid_argument("Bean Player not properly initialized");
        }
        break;
      case PlayerType::zwikPlayer:
        if (zwik_helper != nullptr) {
          manager_mutex.lock();
          players[player_i] = zwik_helper->ai_total_players[id][zwik_player_i];
          manager_mutex.unlock();
          players[player_i]->activated = true;
          game->assigned_players[player_i] = true;
          ++zwik_player_i;
        }
        else {
          throw std::invalid_argument("Zwik Player not properly initialized");
        }
        break;
      case PlayerType::consolePlayer:
        throw std::invalid_argument("Console Player not properly initialized");
        break;
      case PlayerType::guiPlayer:
        players[player_i] = new Player(&game->board, index_color(player_i));
        players[player_i]->agent = new GuiPlayer(players[player_i]);
        players[player_i]->activated = true;
        game->assigned_players[player_i] = true;
        break;
      case PlayerType::randomPlayer:
        players[player_i] = new Player(&game->board, index_color(player_i));
        players[player_i]->agent = new RandomPlayer(players[player_i], new_seed(gen));
        players[player_i]->activated = true;
        game->assigned_players[player_i] = true;
        break;
      case PlayerType::NoPlayer:
        break;
    }
  }
  game->add_players(players);
}

void GameManager::run() {
  clock_t begin_clock = clock();

  if (!updating) {
    if (ready_for_update) {
      update_ai();
    }
    ready_for_update = false;

    assign_players();

    if (log.type == MoveLog || log.type == BothLogs) {
      Move move;
      move.type = MoveType::Replay;
      move.index = 0;
      log.moves[0] = move;
      ++log.writes;
    }

    game->run_game();

    if (log.type == GameLog || log.type == BothLogs) {
      add_game_to_log();
    }

    write_log_to_disk();
    log.writes = 0;

    if (log.type != NoLog) {
      if (seed == 0) {
        game->reseed(new_seed(gen));
      }
      else {
        game->reseed(seed);
      }
    }

    update_ai();

    game->reset();

    ++total_games_played;
    ++games_played;
  }
  else {
    ready_for_update = true;
  }

  clock_t end_clock = clock();
  run_speed = (double)(end_clock - begin_clock) / CLOCKS_PER_SEC;
}

void GameManager::run_single_game() {
  game->log = &log;
  game->reseed(seed);
  update_ai();

  run();

  write_log_to_disk();
  close_log();
}

void GameManager::run_multiple_games() {
  cudaError_t err = cudaStreamCreate(&cuda_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
  }

  game->log = &log;
  game->reseed(seed);
  update_ai();

  while(keep_running) {
    run();
  }

  write_log_to_disk();
  close_log();
}

