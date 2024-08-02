#include "window_player.h"
#include "iostream"

bool show_all_moves;
bool disable_end_turn[4] = {true, true, true, true};
int current_move[4] = {0, 0, 0, 0};
int current_structure[4] = {0, 0, 0, 0};
Move moves[4] = {Move(), Move(), Move(), Move()};

void CheckAvailableTypes(Game* game, int player_id) {
  bool street, village, city = false;
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {
    if (game->players[player_id]->available_moves[move_i].move_type == NoMove) {
      break;
    }
    switch (game->players[player_id]->available_moves[move_i].move_type) {
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
  PlayerState player_state = game->players[player_id]->agent->get_player_state();
  if (player_state == Playing) {
    CheckAvailableTypes(game, player_id);
    if (game->game_state == PlayingRound) {
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
      game->gui_moves[player_id] = moves[player_id];
      game->human_input_received();
      CheckAvailableTypes(game, player_id);
    }
    if (disable_end_turn[player_id]) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    ImGui::EndTable();
  }


  if (game->players[player_id]->agent->get_player_type() == PlayerType::guiPlayer) {
    ImGui::Combo("Move", &current_move[player_id], "Build\0Trade\0\0");

    switch (current_move[player_id]) {
      case 0:
        int hovered_row = -1;

        // Build
        if (current_structure[player_id] != 3) {
          ImGui::Combo("Structure", &current_structure[player_id], "Street\0Village\0City\0\0");

          if (player_id == game->current_player_id) {
            if (ImGui::BeginTable("table_advanced", 2)) {
              ImGui::TableSetupColumn("Corner", ImGuiTableColumnFlags_WidthFixed, 50.0f);
              ImGui::TableSetupColumn("Action");
              ImGui::TableHeadersRow();

              ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
              for (int move_i = 0; move_i < max_available_moves; ++move_i) {
                if (game->players[player_id]->available_moves[move_i].move_type == NoMove && !show_all_moves) {
                  break;
                }
                if (game->players[player_id]->available_moves[move_i].move_type == index_move(current_structure[player_id])) {
                  ImGui::TableNextColumn(); ImGui::Text("%i", game->players[player_id]->available_moves[move_i].index);
                  ImGui::TableNextColumn();

                  // Build Button
                  char button_label[8];
                  sprintf(button_label, "Build##%i", move_i);
                  if (ImGui::SmallButton(button_label)) {
                    moves[player_id].move_type = index_move(current_structure[player_id]);
                    moves[player_id].index = game->players[player_id]->available_moves[move_i].index;

                    game->gui_moves[player_id] = moves[player_id];
                    game->human_input_received();
                    CheckAvailableTypes(game, player_id);
                  }
                  ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
                }
                if (ImGui::IsItemHovered()) {
                  hovered_row = game->players[player_id]->available_moves[move_i].index;
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
        ImGui::TableNextColumn(); ImGui::Text(card_names_char[card_type_i]);
        ImGui::TableNextColumn(); ImGui::Text("%i", game->players[player_id]->cards[card_type_i]);
      }

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
        if (game->players[player_id]->available_moves[move_i].move_type == NoMove && !show_all_moves) {
          break;
        }
        ImGui::TableNextColumn(); ImGui::Text("%i", move_i + 1);
        ImGui::TableNextColumn(); ImGui::Text(move2string(game->players[player_id]->available_moves[move_i]).c_str());
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 1);
      }

      ImGui::EndTable();
    }
  }

}
