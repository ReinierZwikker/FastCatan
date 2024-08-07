#include "window_player.h"
#include "iostream"
#include <mutex>

std::mutex player_mutex;

bool show_all_moves;
bool disable_end_turn[4] = {true, true, true, true};
int current_move[4] = {0, 0, 0, 0};
int current_structure[4] = {0, 0, 0, 0};
Move moves[4] = {Move(), Move(), Move(), Move()};

void CheckAvailableTypes(Game* game, int player_id) {
  bool street, village, city = false;
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {

    player_mutex.lock();
    MoveType move_type = game->players[player_id]->available_moves[move_i].move_type;
    player_mutex.unlock();

    if (move_type == NoMove) {
      break;
    }
    switch (move_type) {
      case buildStreet:
        street = true;
      case buildVillage:
        village = true;
      case buildCity:
        city = true;
    }

    if (street && village && city) {
      break;
    }
  }
  if (street || village || city) {
    current_structure[player_id] = 0;
  }

  if (current_structure[player_id] == 0) {
    if (!street) {
      ++current_structure[player_id];
    }
  }
  if (current_structure[player_id] == 1) {
    if (!village) {
      ++current_structure[player_id];
    }
  }
  if (current_structure[player_id] == 2) {
    if (!city) {
      ++current_structure[player_id];
    }
  }
}


void WindowPlayer(Game* game, ViewPort* viewport, int player_id) {
  player_mutex.lock();
  PlayerState player_state = game->players[player_id]->agent->get_player_state();
  GameStates game_state = game->game_state;
  player_mutex.unlock();

  if (player_state == Playing) {
    CheckAvailableTypes(game, player_id);
    if (game_state == PlayingRound) {
      disable_end_turn[player_id] = false;
    }
  }
  else {
    disable_end_turn[player_id] = true;
  }

  if (ImGui::BeginTable("split", 3)) {
    ImGui::TableNextColumn(); ImGui::Text("Status");
    ImGui::TableNextColumn(); ImGui::Text("%s", player_state_char[player_state]);

    // End Turn Button
    if (disable_end_turn[player_id]) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    ImGui::TableNextColumn(); ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("End Turn").x);
    if (ImGui::Button("End Turn")) {
      moves[player_id] = Move();
      moves[player_id].move_type = endTurn;

      player_mutex.lock();
      game->gui_moves[player_id] = moves[player_id];
      game->human_input_received();
      player_mutex.unlock();

      CheckAvailableTypes(game, player_id);
    }
    if (disable_end_turn[player_id]) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::EndTable();
  }

  player_mutex.lock();
  PlayerType player_type = game->players[player_id]->agent->get_player_type();
  player_mutex.unlock();

  if (player_type == PlayerType::guiPlayer) {
    ImGui::Combo("Move", &current_move[player_id], "Build\0Trade\0\0");

    switch (current_move[player_id]) {
      case 0:
        int hovered_row = -1;

        // Build
        if (current_structure[player_id] != 3) {
          ImGui::Combo("Structure", &current_structure[player_id], "Street\0Village\0City\0\0");

          player_mutex.lock();
          int current_player_id = game->current_player_id;
          player_mutex.unlock();

          if (player_id == current_player_id) {
            if (ImGui::BeginTable("table_advanced", 2)) {
              ImGui::TableSetupColumn("Corner", ImGuiTableColumnFlags_WidthFixed, 50.0f);
              ImGui::TableSetupColumn("Action");
              ImGui::TableHeadersRow();

              ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
              for (int move_i = 0; move_i < max_available_moves; ++move_i) {
                player_mutex.lock();
                MoveType move_type = game->players[player_id]->available_moves[move_i].move_type;
                int move_index_id = game->players[player_id]->available_moves[move_i].index;
                player_mutex.unlock();

                if (move_type == NoMove && !show_all_moves) {
                  break;
                }
                if (move_type == index_move(current_structure[player_id])) {
                  ImGui::TableNextColumn(); ImGui::Text("%i", move_index_id);
                  ImGui::TableNextColumn();

                  // Build Button
                  char button_label[8];
                  sprintf(button_label, "Build##%i", move_i);
                  if (ImGui::SmallButton(button_label)) {

                    player_mutex.lock();
                    moves[player_id].move_type = index_move(current_structure[player_id]);
                    moves[player_id].index = game->players[player_id]->available_moves[move_i].index;

                    game->gui_moves[player_id] = moves[player_id];
                    game->human_input_received();
                    player_mutex.unlock();

                    CheckAvailableTypes(game, player_id);
                  }
                  ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
                }
                if (ImGui::IsItemHovered()) {
                  hovered_row = move_index_id;
                }
              }
              ImGui::EndTable();
            }
            if (hovered_row != -1) {
              if (current_structure[player_id] == 1 || current_structure[player_id] == 2) {
                viewport->player_corner_selection_item.id = hovered_row;
                viewport->player_corner_selection_item.game = game;
                viewport->player_corner_selection_item.render = true;
                if (current_structure[player_id] == 1) {
                  viewport->player_corner_selection_item.corner_occupancy = Village;
                }
                else {
                  viewport->player_corner_selection_item.corner_occupancy = City;
                }
              }
              else {
                viewport->player_street_selection_item.id = hovered_row;
                viewport->player_street_selection_item.game = game;
                viewport->player_street_selection_item.render = true;
              }
            }
          }
        }

        break;
    }

  }

  if (ImGui::CollapsingHeader("Cards")) {
    if (ImGui::BeginTable("split", 2)) {
      ImGui::TableNextColumn(); ImGui::Text("Resource");
      ImGui::TableNextColumn(); ImGui::Text("Amount");
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      for (int card_type_i = 0; card_type_i < 5; card_type_i++) {
        player_mutex.lock();
        int player_card = game->players[player_id]->cards[card_type_i];
        player_mutex.unlock();

        ImGui::TableNextColumn(); ImGui::Text(card_names_char[card_type_i]);
        ImGui::TableNextColumn(); ImGui::Text("%i", player_card);
      }

      ImGui::EndTable();
    }
  }

  if (ImGui::CollapsingHeader("Resources")) {
    if (ImGui::BeginTable("split", 2)) {
      player_mutex.lock();
      int resources_left_0 = game->players[player_id]->resources_left[0];
      int resources_left_1 = game->players[player_id]->resources_left[1];
      int resources_left_2 = game->players[player_id]->resources_left[2];
      player_mutex.unlock();

      ImGui::TableNextColumn(); ImGui::Text("Resource");
      ImGui::TableNextColumn(); ImGui::Text("Amount");
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      ImGui::TableNextColumn(); ImGui::Text("Streets");
      ImGui::TableNextColumn(); ImGui::Text("%i", resources_left_0);
      ImGui::TableNextColumn(); ImGui::Text("Villages");
      ImGui::TableNextColumn(); ImGui::Text("%i", resources_left_1);
      ImGui::TableNextColumn(); ImGui::Text("Cities");
      ImGui::TableNextColumn(); ImGui::Text("%i", resources_left_2);

      ImGui::EndTable();
    }
  }

  if (ImGui::CollapsingHeader("Possible Moves")) {
    ImGui::Checkbox("Show All Moves", &show_all_moves);

    if (ImGui::BeginTable("split", 2, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed, 30.0f);
      ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      for (int move_i = 0; move_i < max_available_moves; move_i++) {
        player_mutex.lock();
        MoveType move_type = game->players[player_id]->available_moves[move_i].move_type;
        player_mutex.unlock();

        if (move_type == NoMove && !show_all_moves) {
          break;
        }

        player_mutex.lock();
        Move move = game->players[player_id]->available_moves[move_i];
        player_mutex.unlock();

        ImGui::TableNextColumn(); ImGui::Text("%i", move_i + 1);
        ImGui::TableNextColumn(); ImGui::Text(move2string(move).c_str());
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      }

      ImGui::EndTable();
    }
  }

}
