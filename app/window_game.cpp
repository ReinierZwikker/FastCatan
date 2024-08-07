#include "window_game.h"
#include <thread>
#include <mutex>
#include <iostream>
#include <string>

int current_player = 0;
std::string current_color;
bool keep_running = false;
bool run = false;
int num_games = 0;

void WindowGame(Game* game) {
  std::thread start_thread;
  std::thread step_round_thread;
  std::thread game_thread;

  std::mutex mutex;

  current_player = game->current_player_id;

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
    ImGui::TableNextColumn(); ImGui::Text("Winning Player:");
    if (game->game_winner == NoColor) {
      ImGui::TableNextColumn(); ImGui::Text("Game in progress...");
    } else {
      ImGui::TableNextColumn(); ImGui::Text("%i - %s", color_index(game->game_winner), color_name(game->game_winner).c_str());
    }
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
    for (int player_i = 0; player_i < 4; ++player_i) {
      ImGui::TableNextColumn(); ImGui::Text("%s = %i VP", color_name(index_color(player_i)).c_str(), game->players[player_i]->victory_points);
    }
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

//  if (game->game_state == SetupRoundFinished || game->game_state == RoundFinished) {
//    game->game_state = PlayingRound;
//    step_round_thread = std::thread(&Game::step_round, game);
//    step_round_thread.detach();
//  }

  // Start game button
  if (game->game_state != ReadyToStart) {
    ImGui::BeginDisabled(true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Run Full Game")) {
    game_thread = std::thread(&Game::run_game, game);
    game_thread.detach();
  }
  if (game->game_state != ReadyToStart) {
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
  }

  if (ImGui::Button("Reset")) {
    game->reset();
  }

  ImGui::Checkbox("Keep Running", &keep_running);


  if (ImGui::Button("Run Multiple Games")) {
    run = true;
  }


  if (run) {
    if (keep_running) {
      if (game->game_state == ReadyToStart) {
        game_thread = std::thread(&Game::run_game, game);
        game_thread.detach();
        ++num_games;
        std::cout << num_games << std::endl;
      }
      else if (game->game_state == GameFinished){
        game->reset();
      }
    }
    else {
      run = false;
      num_games = 0;
    }
  }

}
