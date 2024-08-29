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

  // Check if all threads are done with their epoch
  bool epoch_finished = true;
  if (app_info->state == AppState::Training) {
    for (int game_i = 0; game_i < num_threads; game_i++) {
      if (game_managers[game_i].keep_running) {
        epoch_finished = false;
      }
    }
  } else {
    epoch_finished = false;
  }

  // Disable button if currently training
  if (app_info->state == AppState::Training) {
    blocking_button = true;
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }

  // Stop if this was the last epoch
  if (epoch_finished && !continue_after_epoch) {
    app_info->state = AppState::Idle;
  }

  bool training_button = ImGui::Button("Train", ImVec2(-1.0f, 0.0f));

  // Set up first training (Only first time)
  if (training_button) {
    app_info->num_players = count_players(app_info->selected_players);

    for (int game_i = 0; game_i < num_threads; game_i++) {
      game_managers[game_i].app_info = *app_info;  // Put the current app_info into the game manager
      game_managers[game_i].epoch_length = set_epoch_length;
      if (bean_helper_active) { game_managers[game_i].add_ai_helper(bean_helper); }
      if (zwik_helper_active) { game_managers[game_i].add_ai_helper(zwik_helper); }
    }
  }

  if (continue_after_epoch && epoch_finished) {
    if (zwik_helper_active) {
      zwik_helper->update_epoch();
    }
  }

  // Start training (Start of each epoch)
  if (training_button || (epoch_finished && continue_after_epoch)) {
    for (int game_i = 0; game_i < num_threads; game_i++) {
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
      bean_helper = new BeanHelper(bean_pop_size, (uint8_t)layers,
                                   (uint16_t)nodes_per_layer, (unsigned int)bean_seed, processor_count);
    }
    ImGui::EndTable();
  }

  if (ImGui::BeginTable("bean_settings", 1)) {
    ImGui::TableNextColumn();
    ImGui::SliderInt("Population Size", &bean_pop_size, 4, 2000);

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
  if (ImGui::BeginTable("zwikstatus", 3)) {
    if (zwik_helper == nullptr) {
      ImGui::TableNextColumn(); ImGui::Text("Status");
      ImGui::TableNextColumn(); ImGui::Text("Not Initialized");
    }
    else {
      ImGui::TableNextColumn(); ImGui::Text("Status");
      ImGui::TableNextColumn(); ImGui::Text("Ready");
    }

    ImGui::TableNextColumn();
    if (ImGui::Button("Initialize", ImVec2(-1.0f, 0.0f))) {
      if (zwik_helper != nullptr) {
        delete zwik_helper;
      }
      zwik_helper_active = true;
      zwik_helper = new ZwikHelper(zwik_pop_size, num_threads);
    }
    ImGui::EndTable();
  }
  if (ImGui::Button("Set all players to zwikPlayer", ImVec2(-1.0f, 0.0f))) {
    app_info->selected_players[0] = PlayerType::zwikPlayer;
    app_info->selected_players[1] = PlayerType::zwikPlayer;
    app_info->selected_players[2] = PlayerType::zwikPlayer;
    app_info->selected_players[3] = PlayerType::zwikPlayer;
  }

  if (ImGui::BeginTable("zwik_settings", 1)) {
    ImGui::TableNextColumn();
    ImGui::SliderInt("Population Size", &zwik_pop_size, 4, 2000);

    ImGui::EndTable();
  }
  if (zwik_helper_active) {
    if (ImGui::BeginTable("zwik_ai_s", 4)) {
      ImGui::TableNextColumn(); ImGui::Text("Index");
      ImGui::TableNextColumn(); ImGui::Text("Age");
      ImGui::TableNextColumn(); ImGui::Text("Score");
      ImGui::TableNextColumn(); ImGui::Text("Store?");

      for (int indiv_i = 0; indiv_i < zwik_helper->population_size; ++indiv_i) {
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
        ImGui::TableNextColumn(); ImGui::Text("%i", zwik_helper->get_gene(indiv_i).id);
        ImGui::TableNextColumn(); ImGui::Text("%i", zwik_helper->get_gene(indiv_i).age);
        ImGui::TableNextColumn(); ImGui::Text("%f", zwik_helper->get_gene(indiv_i).score);
        ImGui::TableNextColumn();
        if (ImGui::Button((std::string("Store##") + std::to_string(indiv_i)).c_str(), ImVec2(-1.0f, 0.0f))) {
          zwik_helper->store_gene(indiv_i,
                                  "ai_" + std::to_string(zwik_helper->get_gene(indiv_i).id),
                                  (std::filesystem::path) "ais");
        }
      }


      ImGui::EndTable();
    }
  } else {
    ImGui::Text("Cannot show table: Not initialised");
  }

}

void WindowAI::show(Game* game, AppInfo* app_information) {
  app_info = app_information;

  int total_games_played = 0;
  std::mutex mutex;

  ImGui::Text("Processor Count = %i", processor_count);

  ImGui::InputInt("Threads", &num_threads);
  if (num_threads > processor_count) {
    num_threads = (int)processor_count;
  }
  else if (num_threads < 1) {
    num_threads = 1;
  }

  ImGui::InputInt("Epoch Size", &set_epoch_length);
  if (set_epoch_length < 0) {
    set_epoch_length = 0;
  }
  if (set_epoch_length == 0) {
    ImGui::Text("Epoch limit disabled");
  }

  ImGui::Checkbox("Continue after finishing epoch?", &continue_after_epoch);

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
    for (int game_i = 0; game_i < num_threads; game_i++) {
      int loaded_games_played = game_managers[game_i].games_played.load();
      ImGui::Text("[Thread %i] Game %i", game_i + 1, loaded_games_played);
      total_games_played += loaded_games_played;
    }

    ImGui::Text("Total games played: %i", total_games_played);
  }
}
