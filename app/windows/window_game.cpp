#include "window_game.h"
#include <thread>
#include <mutex>
#include <iostream>
#include <string>

bool keep_running = false;
bool run = false;
int num_games = 0;

void WindowGame(Game* game) {
  std::thread start_thread;
  std::thread step_round_thread;
  std::thread game_thread;

  std::mutex mutex;

  mutex.lock();
  int current_player = game->current_player_id;
  int current_round = game->current_round;
  int num_players = game->num_players;
  int die_1 = game->die_1;
  int die_2 = game->die_2;
  GameStates game_state = game->game_state;
  Color game_winner = game->game_winner;
  mutex.unlock();

  int longest_route, most_knights;
  Color longest_route_color, most_knights_color;
  mutex.lock();
  if (game->longest_road_player != nullptr) {
    longest_route = game->longest_road_player->longest_route;
    longest_route_color = game->longest_road_player->player_color;
  }
  else {
    longest_route = 0;
    longest_route_color = Color::NoColor;
  }
  mutex.unlock();

  mutex.lock();
  if (game->most_knights_player != nullptr) {
    most_knights = game->most_knights_player->played_knight_cards;
    most_knights_color = game->most_knights_player->player_color;
  }
  else {
    most_knights = 0;
    most_knights_color = Color::NoColor;
  }
  mutex.unlock();

  std::string current_color = color_name(index_color(current_player));

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableNextColumn(); ImGui::Text("Game Status:");
    ImGui::TableNextColumn(); ImGui::Text(game_states[game_state]);
    ImGui::TableNextColumn(); ImGui::Text("Round:");
    ImGui::TableNextColumn(); ImGui::Text("%i", current_round);
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::TableNextColumn(); ImGui::Text("Players:");
    ImGui::TableNextColumn(); ImGui::Text("%i", num_players);
    ImGui::TableNextColumn(); ImGui::Text("Current Player:");
    ImGui::TableNextColumn(); ImGui::Text("%i - %s", current_player, current_color.c_str());
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::TableNextColumn(); ImGui::Text("Dice Roll:");
    ImGui::TableNextColumn(); ImGui::Text("%i + %i = %i", die_1, die_2, die_1 + die_2);
    ImGui::TableNextColumn(); ImGui::Text("Winning Player:");
    if (game_winner == Color::NoColor) {
      ImGui::TableNextColumn(); ImGui::Text("Game in progress...");
    } else {
      ImGui::TableNextColumn(); ImGui::Text("%i - %s", color_index(game_winner), color_name(game_winner).c_str());
    }
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
    mutex.lock();
    for (int player_i = 0; player_i < 4; ++player_i) {
      ImGui::TableNextColumn(); ImGui::Text("%s = %i VP", color_name(index_color(player_i)).c_str(), game->players[player_i]->victory_points);
    }
    mutex.unlock();
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::TableNextColumn(); ImGui::Text("Longest Route:");
    ImGui::TableNextColumn(); ImGui::Text("%s - %i", color_names[color_index(longest_route_color)].c_str(), longest_route);
    ImGui::TableNextColumn(); ImGui::Text("Most Knights:");
    ImGui::TableNextColumn(); ImGui::Text("%s - %i", color_names[color_index(most_knights_color)].c_str(), most_knights);
    ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);

    ImGui::EndTable();
  }

  if (ImGui::BeginTable("round_options", 4)) {
    // Setup Round button
    ImGui::TableNextColumn();
    if (game_state != ReadyToStart) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Setup Round", ImVec2(-1.0f, 0.0f))) {
      start_thread = std::thread(&Game::start_game, game);
      start_thread.detach();
    }
    if (game_state != ReadyToStart) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    // Setup Round button
    ImGui::TableNextColumn();
    if (game_state != SetupRoundFinished && game_state != RoundFinished) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Step Round", ImVec2(-1.0f, 0.0f))) {
      game->game_state = PlayingRound;
      step_round_thread = std::thread(&Game::step_round, game);
      step_round_thread.detach();
    }
    if (game_state != SetupRoundFinished && game_state != RoundFinished) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    // Run Full Game Button
    ImGui::TableNextColumn();
    if (game_state != ReadyToStart) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    if (ImGui::Button("Run Full Game", ImVec2(-1.0f, 0.0f))) {
      game_thread = std::thread(&Game::run_game, game);
      game_thread.detach();
    }
    if (game_state != ReadyToStart) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    // Reset Game Button
    ImGui::TableNextColumn();
    if (ImGui::Button("Reset", ImVec2(-1.0f, 0.0f))) {
      game->reset();
      //bool reset_finished = false;
      //while(!reset_finished) {
      //  reset_finished = true;
      //  for (auto player: game->players) {
      //    if (player == nullptr) {
      //      reset_finished = false;
      //    }
      //  }
      //}
    }

    ImGui::EndTable();
  }

  if (ImGui::CollapsingHeader("Development Cards")) {
    if (ImGui::BeginTable("split", 1)) {
      mutex.lock();
      int current_development_card = game->current_development_card;
      mutex.unlock();

      ImGui::TableNextColumn(); ImGui::Text("Available: ");
      mutex.lock();
      for (int dev_card_i = 0; dev_card_i < amount_of_development_cards; ++dev_card_i) {
        if (dev_card_i == current_development_card) {
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(255, 0, 0, 128)); // Red color with 50% opacity
        }
        ImGui::TableNextColumn(); ImGui::Text("%s", dev_card_names_char[game->development_cards[dev_card_i]]);
      }
      mutex.unlock();

      ImGui::EndTable();
    }
  }

//  if (game->game_state == SetupRoundFinished || game->game_state == RoundFinished) {
//    game->game_state = PlayingRound;
//    step_round_thread = std::thread(&Game::step_round, game);
//    step_round_thread.detach();
//  }

  // Start game button

//
//  if (ImGui::Button("Reset")) {
//    game->reset();
//  }
//
//  ImGui::Checkbox("Keep Running", &keep_running);
//
//
//  if (ImGui::Button("Run Multiple Games")) {
//    run = true;
//  }
//
//
//  if (run) {
//    if (keep_running) {
//      if (game_state == ReadyToStart) {
//        game_thread = std::thread(&Game::run_game, game);
//        game_thread.detach();
//        ++num_games;
//        std::cout << num_games << std::endl;
//      }
//      else if (game_state == GameFinished){
//        game->reset();
//      }
//    }
//    else {
//      run = false;
//      num_games = 0;
//    }
//  }

}
