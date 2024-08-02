#include "window_game.h"
#include <thread>
#include <mutex>
#include <iostream>
#include <string>

int current_player = 0;
std::string current_color;

void WindowGame(Game* game) {
  std::thread start_thread;
  std::thread step_round_thread;

  std::mutex mutex;

  mutex.lock();
  current_player = game->current_player_id;
  mutex.unlock();

  current_color = color_name(index_color(current_player));

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableNextColumn(); ImGui::Text("Game Status:");
    ImGui::TableNextColumn(); ImGui::Text(game_states[game->game_state]);
    ImGui::TableNextColumn(); ImGui::Text("Round:");
    ImGui::TableNextColumn(); ImGui::Text("%i", game->current_round);
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::TableNextColumn(); ImGui::Text("Players:");
    ImGui::TableNextColumn(); ImGui::Text("%i", game->num_players);
    ImGui::TableNextColumn(); ImGui::Text("Current Player:");
    ImGui::TableNextColumn(); ImGui::Text("%i - %s", current_player, current_color.c_str());
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableNextColumn(); ImGui::Text("Dice Roll:");
    ImGui::TableNextColumn(); ImGui::Text("%i + %i = %i", game->die_1, game->die_2, game->die_1 + game->die_2);
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::EndTable();
  }

  // Start game button
  if (game->game_state != ReadyToStart) {
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Start Game")) {
    start_thread = std::thread(&Game::start_game, game);
    start_thread.detach();
  }
  if (game->game_state != ReadyToStart) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }

  if (game->game_state == SetupRoundFinished || game->game_state == RoundFinished) {
    std::cout << "created thread" << std::endl;
    step_round_thread = std::thread(&Game::step_round, game);
    step_round_thread.detach();
    game->game_state = PlayingRound;
  }

}
