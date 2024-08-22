#include "window_ai.h"

static char folder[50] = "logs";
static const std::string save_name = "/GameLog_Thread_";

WindowAI::WindowAI() {

}

WindowAI::~WindowAI() {

}

int count_players (PlayerType* players) {
  uint8_t num_players = 0;
  for (int player_i = 0; player_i < 4; ++player_i) {
    if (players[player_i] != PlayerType::NoPlayer) {
      ++num_players;
    }
  }

  return num_players;
}

void WindowAI::train_button() {
  bool blocking_button = false;
  if (app_info->state == AppState::Training) {
    blocking_button = true;
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Train", ImVec2(-1.0f, 0.0f))) {
    app_info->num_players = count_players(app_info->selected_players);

    for (int game_i = 0; game_i < num_threads; game_i++) {
      game_managers[game_i].app_info = *app_info;  // Put the current app_info into the game manager
      if (bean_helper_active) { game_managers[game_i].add_ai_helper(bean_helper); }
      if (zwik_helper_active) { game_managers[game_i].add_ai_helper(zwik_helper); }

      game_managers[game_i].keep_running = true;
      game_managers[game_i].id = game_i;

      game_managers[game_i].log.type = static_cast<LogType>(log_type);
      game_managers[game_i].start_log(game_managers[game_i].log.type,
                                      std::string(folder) + save_name + std::to_string(game_i + 1), folder);

      game_managers[game_i].add_seed(seed);
      threads[game_i] = std::thread(&GameManager::run_multiple_games, &game_managers[game_i]);
      threads[game_i].detach();
    }
    app_info->state = AppState::Training;
  }
  if (blocking_button) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }
}

void WindowAI::stop_training_button() {
  bool blocking_button = false;
  if (app_info->state != AppState::Training) {
    blocking_button = true;
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Stop Training", ImVec2(-1.0f, 0.0f))) {
    for (int game_i = 0; game_i < num_threads; game_i++) {
      game_managers[game_i].keep_running = false;
      game_managers[game_i].close_log();
    }
    app_info->state = AppState::Idle;
  }
  if (blocking_button) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }
}

void WindowAI::select_players_window() {
  for (int player_i = 0; player_i < 4; ++player_i) {
    int selected_player_type = (int)app_info->selected_players[player_i];
    std::string label = "[" + std::to_string(player_i) + "] Player Type";
    ImGui::Combo(label.c_str(), &selected_player_type,
                 "Console Player\0GUI Player\0Random Player\0Zwik Player\0Bean Player\0No Player\0\0");
    if (selected_player_type == (int)PlayerType::consolePlayer ||
        selected_player_type == (int)PlayerType::guiPlayer ||
        selected_player_type == (int)PlayerType::NoPlayer) {
      show_player_error[player_i] = 255;
    }
    else {
      app_info->selected_players[player_i] = static_cast<PlayerType>(selected_player_type);
    }

    if (show_player_error[player_i] != 0) {
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
      ImGui::Text("Player Type is not available");
      ImGui::PopStyleColor();
      --show_player_error[player_i];
    }
  }
}

void WindowAI::bean_ai_window(Game* game) {
  if (ImGui::BeginTable("beanstatus", 3)) {
    if (bean_helper == nullptr) {
      ImGui::TableNextColumn(); ImGui::Text("Status");
      ImGui::TableNextColumn(); ImGui::Text("Not Initialized");
    }
    else {
      ImGui::TableNextColumn(); ImGui::Text("Status");
      ImGui::TableNextColumn(); ImGui::Text("Ready");
    }

    ImGui::TableNextColumn();
    if (ImGui::Button("Initialize", ImVec2(-1.0f, 0.0f))) {
      if (bean_helper != nullptr) {
        delete bean_helper;
      }
      bean_helper_active = true;
      bean_helper = new BeanHelper(bean_pop_size, (unsigned int)bean_seed, processor_count);
      for (int player_i = 0; player_i < app_info->num_players; ++player_i) {
        app_info->selected_players[player_i] = PlayerType::beanPlayer;
      }

    }
    ImGui::EndTable();
  }

  if (ImGui::BeginTable("bean_settings", 1)) {
    ImGui::TableNextColumn();
    ImGui::SliderInt("Population Size", &bean_pop_size, (int)processor_count * 4, 2000);

    ImGui::TableNextColumn();
    ImGui::SliderInt("Amount of layers", &layers, 1, 50);

    ImGui::TableNextColumn();
    ImGui::SliderInt("Nodes per layer", &nodes_per_layer, 1, 300);

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("bean_seed", 2)) {
    ImGui::TableNextColumn();
    ImGui::InputInt("Seed", &bean_seed, 0);

    ImGui::TableNextColumn();
    ImGui::Checkbox("Randomize Seed", &randomize_seed);

    ImGui::EndTable();
  }

}

void WindowAI::zwik_ai_window(Game* game) {

}

void WindowAI::thread_table(Game* game) {
  static ImGuiTableFlags flags = ImGuiTableFlags_RowBg |
                                 ImGuiTableFlags_Resizable |
                                 ImGuiTableFlags_Reorderable |
                                 ImGuiTableFlags_ScrollY;

  ImGui::Text("Total games played: %i", total_games_played);
  total_games_played = 0;

  if (ImGui::BeginTable("thread_table", 11, flags)) {
    ImGui::TableSetupColumn("Thread", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableSetupColumn("Game N", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableSetupColumn("Avg Turns", 50.0f);
    ImGui::TableSetupColumn("G Win%",    50.0f);
    ImGui::TableSetupColumn("G Avg VP",  50.0f);
    ImGui::TableSetupColumn("R Win%",    50.0f);
    ImGui::TableSetupColumn("R Avg VP",  50.0f);
    ImGui::TableSetupColumn("W Win%",    50.0f);
    ImGui::TableSetupColumn("W Avg VP",  50.0f);
    ImGui::TableSetupColumn("B Win%",    50.0f);
    ImGui::TableSetupColumn("B Avg VP",  50.0f);
    ImGui::TableHeadersRow();

    for (int thread = 0; thread < num_threads; thread++) {
      int loaded_games_played = game_managers[thread].total_games_played.load();

      ImGui::TableNextColumn(); ImGui::Text("%i", thread + 1);
      ImGui::TableNextColumn(); ImGui::Text("%i", loaded_games_played);
      if (bean_helper != nullptr) {
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][0].average_rounds);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][0].win_rate);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][0].average_points);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][1].win_rate);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][1].average_points);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][2].win_rate);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][2].average_points);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][3].win_rate);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->ai_total_summaries[thread][3].average_points);
      }
      else {
        ImGui::TableNextRow();
      }

      total_games_played += loaded_games_played;
    }

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("best_players", 6))
  {
    ImGui::TableSetupColumn("Player", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableSetupColumn("Wins", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableSetupColumn("Avg Rounds", 50.0f);
    ImGui::TableSetupColumn("Avg Points", 50.0f);
    ImGui::TableSetupColumn("G Win%",     50.0f);
    ImGui::TableSetupColumn("Score",      50.0f);
    ImGui::TableHeadersRow();

    for (int player_i = 0; player_i < 3; player_i++) {
      if (bean_helper != nullptr) {
        ImGui::TableNextColumn(); ImGui::Text("%i", bean_helper->top_players[player_i].id);
        ImGui::TableNextColumn(); ImGui::Text("%i", bean_helper->top_players[player_i].wins);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->top_players[player_i].average_rounds);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->top_players[player_i].average_points);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->top_players[player_i].win_rate);
        ImGui::TableNextColumn(); ImGui::Text("%.2f", bean_helper->top_player_scores[player_i]);
      }
    }

    ImGui::EndTable();
  }



  if (bean_helper != nullptr) {
    if (total_games_played > 200 * (bean_updates + 1)) {

      bool all_ready = true;
      for (int i = 0; i < num_threads; ++i) {
        game_managers[i].updating = true;
        if (!game_managers[i].ready_for_update) {
          all_ready = false;
        }
      }

      if (all_ready) {
        std::cout << "ready" << std::endl;
        bean_helper->shuffle_players();

        if (total_games_played > 1000 * (bean_evolutions + 1)) {
          std::cout << "evolution" << std::endl;
          bean_helper->eliminate();
          bean_helper->reproduce();
          bean_helper->mutate();

          for (int i = 0; i < num_threads; ++i) {
            game_managers[i].games_played = 0;
          }

          ++bean_evolutions;
        }

        ++bean_updates;

        for (int i = 0; i < num_threads; ++i) {
          game_managers[i].updating = false;
        }
      }
    }
  }

}

void WindowAI::show(Game* game, AppInfo* app_information) {
  app_info = app_information;

  std::mutex mutex;

  ImGui::Text("Processor Count = %i", processor_count);

  ImGui::InputInt("Threads", &num_threads);
  if (num_threads > processor_count) {
    num_threads = (int)processor_count;
  }
  else if (num_threads < 1) {
    num_threads = 1;
  }

  if (ImGui::BeginTable("training_table", 3)) {
    ImGui::TableNextColumn();
    train_button();

    ImGui::TableNextColumn();
    stop_training_button();

    ImGui::TableNextColumn();
    if (ImGui::Button("Select Players")) {
      show_select_players_menu = !show_select_players_menu;
    }

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("log", 2)) {
    ImGui::TableNextColumn();
    ImGui::Combo("Log Type", &log_type, "No Logging\0Move Log\0Game Log\0Both\0\0");

    ImGui::TableNextColumn();
    ImGui::InputText("Folder", folder, 50);

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("AI menus", 2)) {
    ImGui::TableNextColumn();
    if (ImGui::Button("Open Bean AI", ImVec2(-1.0f, 0.0f))) {
      show_bean_ai_menu = !show_bean_ai_menu;
    }
    ImGui::TableNextColumn();
    if (ImGui::Button("Open Zwik AI", ImVec2(-1.0f, 0.0f))) {
      show_zwik_ai_menu = !show_zwik_ai_menu;
    }

    ImGui::EndTable();
  }

  if (show_select_players_menu) {
    ImGui::Begin("Select Players Menu", &show_select_players_menu);
    select_players_window();
    ImGui::End();
  }

  if (show_bean_ai_menu) {
    ImGui::Begin("Bean AI Menu", &show_bean_ai_menu);
    bean_ai_window(game);
    ImGui::End();
  }

  if (show_zwik_ai_menu) {
    ImGui::Begin("Zwik AI Menu", &show_zwik_ai_menu);
    zwik_ai_window(game);
    ImGui::End();
  }

  if (app_info->state == AppState::Training) {
    thread_table(game);
  }
}
