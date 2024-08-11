#include "window_ai.h"

static char folder[50] = "logs";

WindowAI::WindowAI() {

}

WindowAI::~WindowAI() {

}

bool WindowAI::show() {
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

  if (ImGui::BeginTable("split", 2)) {
    ImGui::TableNextColumn();
    if (training) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Train")) {
      for (int game_i = 0; game_i < num_threads; game_i++) {
        game_managers[game_i].keep_running = true;
        game_managers[game_i].id = game_i;

        PlayerType player_type[4] = {PlayerType::randomPlayer, PlayerType::randomPlayer,
                                     PlayerType::randomPlayer, PlayerType::randomPlayer};
        game_managers[game_i].game.add_players(player_type);

        game_managers[game_i].log.type = static_cast<LogType>(log_type);
        game_managers[game_i].start_log(game_managers[game_i].log.type,
                                        std::string(folder) + "/GameLog_Thread_" + std::to_string(game_i + 1), folder);

        game_managers[game_i].seed = seed;
        threads[game_i] = std::thread(&GameManager::run_multiple_games, &game_managers[game_i]);
        threads[game_i].detach();
      }
      do_training = true;
    }
    if (training) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::TableNextColumn();
    if (!training) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Stop Training")) {
      for (int game_i = 0; game_i < num_threads; game_i++) {
        game_managers[game_i].keep_running = false;
        game_managers[game_i].close_log();
      }
      do_training = false;
    }
    if (!training) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::TableNextColumn();
    ImGui::Combo("Log Type", &log_type, "No Logging\0Move Log\0Game Log\0Both\0\0");

    ImGui::TableNextColumn();
    ImGui::InputText("Folder", folder, 50);

    ImGui::EndTable();
  }

  if (do_training) {
    training = true;
    for (int game_i = 0; game_i < num_threads; game_i++) {
      int loaded_games_played = game_managers[game_i].games_played.load();
      ImGui::Text("[Thread %i] Game %i", game_i + 1, loaded_games_played);
      total_games_played += loaded_games_played;
    }

    ImGui::Text("Total games played: %i", total_games_played);
  }
  else {
    training = false;
  }

  return training;
}
