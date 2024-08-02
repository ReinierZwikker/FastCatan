#include "window_game.h"
#include <thread>
#include <mutex>
#include <iostream>

void WindowGame(Game* game) {
  std::thread start_thread;
  std::thread step_round_thread;

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableNextColumn(); ImGui::Text("Game Status:");
    ImGui::TableNextColumn(); ImGui::Text(game_states[game->game_state]);
    ImGui::TableNextColumn(); ImGui::Text("Players:");
    ImGui::TableNextColumn(); ImGui::Text("%i", game->num_players);
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("split", 2)) {
    ImGui::TableNextColumn();
    if (ImGui::Button("Start Game")) {
      start_thread = std::thread(&Game::start_game, game);
      start_thread.detach();
    }
    ImGui::TableNextColumn();
    if (ImGui::Button("Start Step Round")) {
      step_round_thread = std::thread(&Game::step_round, game);
      step_round_thread.detach();
    }

    ImGui::EndTable();
  }
}
