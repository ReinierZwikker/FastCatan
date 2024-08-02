#include "window_game.h"
#include <thread>
#include <mutex>
#include <iostream>

void WindowGame(Game* game) {
  // std::thread start_thread;

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableNextColumn(); ImGui::Text("Game Status:");
    ImGui::TableNextColumn(); ImGui::Text(game_states[game->game_state]);
    ImGui::TableNextColumn(); ImGui::Text("Players:");
    ImGui::TableNextColumn(); ImGui::Text("%i", game->num_players);
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::EndTable();
  }

  if (ImGui::Button("Start")) {
    // start_thread = std::thread(&Game::start_game, game);
  }

  if (ImGui::Button("Resume")) {
    // game->human_input_received();
  }

//  if (start_thread.joinable()) {
//    start_thread.join();
//  }

}
