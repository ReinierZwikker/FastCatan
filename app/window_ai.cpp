#include "window_ai.h"
#include <thread>
#include <iostream>
#include <mutex>

bool do_training = false;
bool training = false;
const unsigned int processor_count = std::thread::hardware_concurrency();
int num_threads = 30;

int games_played[50];
Game games[50];
std::thread threads[50];

bool WindowAI() {
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

  if (training) {
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Train")) {
    for (int game_i = 0; game_i < num_threads; game_i++) {
      threads[game_i] = std::thread(&Game::run_multiple_games, &games[game_i]);
      threads[game_i].detach();
    }
    do_training = true;
  }
  if (training) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }

  if (do_training) {
    training = true;
    for (int game_i = 0; game_i < num_threads; game_i++) {
      int loaded_games_played = games[game_i].games_played.load();
      ImGui::Text("[Thread %i] Game %i", game_i + 1, loaded_games_played);
      total_games_played += loaded_games_played;
    }

    ImGui::Text("Total games played: %i", total_games_played);
  }

  return training;
}
