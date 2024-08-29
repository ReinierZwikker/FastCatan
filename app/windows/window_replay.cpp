#include "window_replay.h"

static std::string input_folder = "logs";
std::thread game_thread;

std::vector<std::string> split (const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss (s);
  std::string item;

  while (getline (ss, item, delim)) {
    result.push_back (item);
  }

  return result;
}

void WindowReplay::reset_replay_state(Game* game, AppInfo* app_info) {
  app_info->state = AppState::Idle;
  play = false;
  play_tick = 0;
  current_move = 1;
  mutex.lock();
  game->reset();
  mutex.unlock();
}

void WindowReplay::input_folder_box() {
  char* input_folder_char = const_cast<char *>(input_folder.c_str());
  ImGui::InputText("Input Folder", input_folder_char, 50);
  input_folder = input_folder_char;
}

void WindowReplay::thread_box() {
  ImGui::InputInt("Thread", &thread_id);
  if (thread_id < 1) {
    thread_id = 1;
  }
  else if (thread_id > processor_count) {
    thread_id = processor_count;
  }
}

void WindowReplay::load_button() {
  if (ImGui::Button("Load", ImVec2(-1.0f, 0.0f))) {
    load_games(input_folder);
  }
}

void WindowReplay::replay_button(Game* game, AppInfo* app_info) {
  bool start_block = false;
  if (loaded_moves.empty()) {
    start_block = true;
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Replay", ImVec2(-1.0f, 0.0f))) {
    if (app_info->state == AppState::Replaying) {
      reset_replay_state(game, app_info);
    }

    mutex.lock();
    AppInfo game_initializer;
    game->num_players = loaded_games[current_game].num_players;
    unsigned int game_seed = loaded_games[current_game].seed;

    game_initializer.num_players = game->num_players;
    for (int player_i = 0; player_i < game->num_players; ++player_i) {
      game_initializer.selected_players[player_i] = PlayerType::guiPlayer;
    }

    game_manager = new GameManager();
    game_manager->app_info = game_initializer;
    game_manager->add_seed(game_seed);
    game_manager->game = game;

    mutex.unlock();

    game_thread = std::thread(&GameManager::run_single_game, game_manager);
    game_thread.detach();

    app_info->state = AppState::Replaying;
  }
  if (start_block) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }
}

void WindowReplay::next_move_button(Game* game, AppInfo* app_info) {
  bool start_block = false;
  if (app_info->state != AppState::Replaying || player_state != Playing || play) {
    start_block = true;
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Next Move", ImVec2(-1.0f, 0.0f))) {
    game->human_input_received(loaded_moves[current_move]);
    ++current_move;
  }
  if (start_block) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }
}

WindowReplay::WindowReplay() {

}

WindowReplay::~WindowReplay() {
  delete game_manager;
}

void WindowReplay::show(Game* game, ViewPort* viewport, AppInfo* app_info) {

  if (app_info->state == AppState::Replaying) {
    mutex.lock();
    game_state = game->game_state;
    player_state = game->current_player->agent->get_player_state();
    mutex.unlock();
  }

  input_folder_box();
  thread_box();

  if (ImGui::BeginTable("file_management", 4, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Load", ImGuiTableColumnFlags_WidthFixed, 70.0f);
    ImGui::TableSetupColumn("Replay", ImGuiTableColumnFlags_WidthFixed, 70.0f);
    ImGui::TableSetupColumn("NextMove", ImGuiTableColumnFlags_WidthFixed, 70.0f);
    ImGui::TableSetupColumn("Messages", ImGuiTableColumnFlags_WidthStretch);

    ImGui::TableNextColumn();
    load_button();

    ImGui::TableNextColumn();
    replay_button(game, app_info);

    ImGui::TableNextColumn();
    next_move_button(game, app_info);

    ImGui::TableNextColumn();
    if (failed_to_load_game) {
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
      ImGui::Text("Failed to load file!");
      ImGui::PopStyleColor();
    }

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("play_menu", 4, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Button1", ImGuiTableColumnFlags_WidthFixed, 70.0f); // Fixed width for Button1
    ImGui::TableSetupColumn("Button2", ImGuiTableColumnFlags_WidthFixed, 70.0f); // Fixed width for Button2
    ImGui::TableSetupColumn("Button3", ImGuiTableColumnFlags_WidthFixed, 70.0f); // Fixed width for Button3
    ImGui::TableSetupColumn("Slider", ImGuiTableColumnFlags_WidthStretch); // Slider takes up remaining space

    ImGui::TableNextColumn();
    bool start_block = false;
    if (app_info->state != AppState::Replaying || play) {
      start_block = true;
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Play", ImVec2(-1.0f, 0.0f))) {
      play = true;
    }
    if (start_block) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::TableNextColumn();
    start_block = false;
    if (app_info->state != AppState::Replaying || !play) {
      start_block = true;
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Pause", ImVec2(-1.0f, 0.0f))) {
      play = false;
    }
    if (start_block) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::TableNextColumn();
    start_block = false;
    if (app_info->state != AppState::Replaying || current_move == 1) {
      start_block = true;
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Stop", ImVec2(-1.0f, 0.0f))) {
      reset_replay_state(game, app_info);
    }
    if (start_block) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::TableNextColumn();
    ImGui::SliderFloat("Playback Speed", &play_speed, 1, std::round(ImGui::GetIO().Framerate));

    ImGui::EndTable();
  }


  if (play) {
    ++play_tick;
    if (play_tick > (unsigned int)(ImGui::GetIO().Framerate / play_speed)) {
      game->human_input_received(loaded_moves[current_move]);
      ++current_move;

      play_tick = 0;
    }
  }

  // Stop the replay
  if ((app_info->state == AppState::Replaying && current_move >= loaded_moves.size() - 1)) {
    play = false;
    play_tick = 0;
  }

  if (ImGui::CollapsingHeader("Games")) {
    if (loaded_games.size() > 0) {
      static ImGuiTableFlags flags = ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_Resizable |
                                     ImGuiTableFlags_Reorderable |
                                     ImGuiTableFlags_ScrollY;
      if (ImGui::BeginTable("games", 9, flags)) {
        ImGui::TableSetupColumn("Game", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Rounds");
        ImGui::TableSetupColumn("Moves Played");
        ImGui::TableSetupColumn("Run Time [ms]");
        ImGui::TableSetupColumn("Winner");
        ImGui::TableSetupColumn("Players");
        ImGui::TableSetupColumn("Seed");
        ImGui::TableSetupColumn("Load");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        for (int game_i = 0; game_i < loaded_games.size(); ++game_i) {
          if (game_i == current_game) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(255, 0, 0, 128)); // Red color with 50% opacity
          }

          ImGui::TableNextColumn(); ImGui::Text("%i", game_i);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].id);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].rounds);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].moves_played);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].run_time);
          ImGui::TableNextColumn(); ImGui::Text("%s", color_names[color_index(loaded_games[game_i].winner)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games[game_i].num_players);
          ImGui::TableNextColumn(); ImGui::Text("%u", loaded_games[game_i].seed);

          ImGui::TableNextColumn();
          char button_label[16];
          sprintf(button_label, "Load##G%i", game_i);
          if (ImGui::SmallButton(button_label)) {
            transfer(input_folder, game_i);
            app_info->state = AppState::Idle;
            play = false;
            play_tick = 0;
            current_game = game_i;
            current_move = 1;

            mutex.lock();
            game->reseed(loaded_games[current_game].seed);
            game->reset();
            viewport->NewMap(game);
            mutex.unlock();
          }

          ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        }
        ImGui::EndTable();
      }
    }
    else {
      load_games(input_folder);
    }
  }

  if (ImGui::CollapsingHeader("Moves")) {
    if (loaded_moves.size() > 0) {
      static ImGuiTableFlags flags = ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_Resizable |
                                     ImGuiTableFlags_Reorderable |
                                     ImGuiTableFlags_ScrollY;
      if (ImGui::BeginTable("moves", 11, flags)) {
        ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Player", ImGuiTableColumnFlags_WidthFixed, 10.0f);
        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 30.0f);
        ImGui::TableSetupColumn("Index");
        ImGui::TableSetupColumn("Other Player");
        ImGui::TableSetupColumn("TX Card");
        ImGui::TableSetupColumn("TX Amount");
        ImGui::TableSetupColumn("RX Card");
        ImGui::TableSetupColumn("RX Amount");
        ImGui::TableSetupColumn("Mistakes");
        ImGui::TableHeadersRow();

        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        int current_player = -1;
        bool first_pass = false;
        bool backwards = false;
        for (int move_i = 0; move_i < loaded_moves.size(); ++move_i) {
          if (move_i == current_move - 1) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(255, 0, 0, 128)); // Red color with 50% opacity
            if (app_info->state == AppState::Replaying && play) {
              float scroll_max_y = ImGui::GetScrollMaxY();
              ImGui::SetScrollY(((float)move_i / (float)loaded_moves.size()) * scroll_max_y);
            }
          }

          ImGui::TableNextColumn(); ImGui::Text("%i", move_i);
          ImGui::TableNextColumn(); ImGui::Text("%s", color_names[current_player].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%s", move2string(loaded_moves[move_i]).c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", move_index(loaded_moves[move_i].type));
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].index);
          ImGui::TableNextColumn(); ImGui::Text("%s", color_names[color_index(loaded_moves[move_i].other_player)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%s", card_names[card_index(loaded_moves[move_i].tx_card)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].tx_amount);
          ImGui::TableNextColumn(); ImGui::Text("%s", card_names[card_index(loaded_moves[move_i].rx_card)].c_str());
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].rx_amount);
          ImGui::TableNextColumn(); ImGui::Text("%i", loaded_moves[move_i].mistakes);

          if (loaded_moves[move_i].type == MoveType::endTurn) {
            if (current_player < loaded_games[current_game].num_players - 1) {
              ++current_player;
            }
            else {
              current_player = 0;
            }
            first_pass = true;
          }
          else if (move_i < (4 * loaded_games[current_game].num_players - 1)) {
            if (first_pass) {
              first_pass = false;
            }
            else {
              first_pass = true;
              if (backwards) {
                --current_player;
              }
              else if (current_player == loaded_games[current_game].num_players - 1) {
                backwards = true;
              }
              else {
                ++current_player;
              }
            }
          }

          ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        }
        ImGui::EndTable();
      }
    }
  }

}

void WindowReplay::load_games(const std::string& folder) {
  std::string path_game = folder + "/GameLog_Thread_" + std::to_string(thread_id) + "_game_summaries.dat";
  FILE* file_game = std::fopen(path_game.c_str(), "rb");

  if (!file_game) {
    failed_to_load_game = true;
    return;
    //throw std::invalid_argument("Could not open file");
  }
  else {
    failed_to_load_game = false;
  }

  // Find the size of the file
  std::fseek(file_game, 0, SEEK_SET);  // Place cursor at the start

  loaded_games.clear();
  GameSummary game_summary;
  while (std::fread(&game_summary, sizeof(GameSummary), 1, file_game) == 1) {
    loaded_games.push_back(game_summary);
  }

  std::fclose(file_game);
}


void WindowReplay::transfer(const std::string& folder, int game_id) {
  std::string path_moves = folder + "/GameLog_Thread_" + std::to_string(thread_id) + "_moves.dat";
  FILE* file_moves = std::fopen(path_moves.c_str(), "rb");

  if (!file_moves) {
    throw std::invalid_argument("Could not open file");
  }

  std::fseek(file_moves, 0, SEEK_SET);  // Place cursor at the start

  loaded_moves.clear();
  for (int game_i = 0; game_i < game_id + 1; ++game_i) {
    for (int move_i = 0; move_i < loaded_games[game_i].moves_played; ++move_i) {
      Move move;
      std::fread(&move, sizeof(Move), 1, file_moves);

      if (game_i == game_id) {
        loaded_moves.push_back(move);
      }
    }
  }

  std::fclose(file_moves);
}

