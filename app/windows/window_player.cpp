#include "window_player.h"
#include "../../src/game/AIPlayer/ai_zwik_player.h"

#include "iostream"
#include <mutex>

std::mutex player_mutex;

bool show_all_moves;
bool disable_end_turn[4] = {true, true, true, true};
int current_move[4] = {0, 0, 0, 0};
int current_structure[4] = {0, 0, 0, 0};
Move moves[4] = {Move(), Move(), Move(), Move()};

char ai_name[10] = "ai_0";


void CheckAvailableTypes(Game* game, int player_id) {
  bool street, village, city = false;
  for (int move_i = 0; move_i < max_available_moves; ++move_i) {

    player_mutex.lock();
    MoveType move_type = game->players[player_id]->available_moves[move_i].type;
    player_mutex.unlock();

    if (move_type == MoveType::NoMove) {
      break;
    }
    switch (move_type) {
      case MoveType::buildStreet:
        street = true;
      case MoveType::buildVillage:
        village = true;
      case MoveType::buildCity:
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


void WindowPlayer(Game* game, ViewPort* viewport, int player_id, AppInfo* app_info) {
  player_mutex.lock();
  PlayerState player_state = game->players[player_id]->agent->get_player_state();
  GameStates game_state = game->game_state;
  player_mutex.unlock();

  if (player_state == Playing && app_info->state != AppState::Replaying) {
    CheckAvailableTypes(game, player_id);
    if (game_state == PlayingRound) {
      disable_end_turn[player_id] = false;
    }
  }
  else {
    disable_end_turn[player_id] = true;
  }

  if (ImGui::BeginTable("split", 4)) {
    ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 90.0f); // Fixed width for Button1
    ImGui::TableSetupColumn("StatusInt", ImGuiTableColumnFlags_WidthStretch, 50.0f); // Fixed width for Button2
    ImGui::TableSetupColumn("PlayerType", ImGuiTableColumnFlags_WidthFixed, 100.0f); // Fixed width for Button3
    ImGui::TableSetupColumn("EndTurn", ImGuiTableColumnFlags_WidthFixed, 50.0f); // Slider takes up remaining space

    ImGui::TableNextColumn(); ImGui::Text("Status");
    ImGui::TableNextColumn(); ImGui::Text("%s", player_state_char[player_state]);

    ImGui::TableNextColumn();
    player_mutex.lock();
    int player_type = (int)game->players[player_id]->agent->get_player_type();
    int current_player_type = (int)game->players[player_id]->agent->get_player_type();  // Save the current player type
    ImGui::Combo("##Player Type", &player_type, "Console  \0GUI      \0Random   \0Zwik     \0Bean     \0No Player\0\0");
    if (current_player_type != player_type) {
      game->add_player((PlayerType)player_type, player_id);
    }
    player_mutex.unlock();

    // End Turn Button
    if (disable_end_turn[player_id]) {
      ImGui::BeginDisabled(true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    }
    ImGui::TableNextColumn(); ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("End Turn").x);
    if (ImGui::Button("End Turn", ImVec2(-1.0f, 0.0f))) {
      moves[player_id] = Move();
      moves[player_id].type = MoveType::endTurn;

      player_mutex.lock();
      game->human_input_received(moves[player_id]);
      player_mutex.unlock();

      CheckAvailableTypes(game, player_id);
    }
    if (disable_end_turn[player_id]) {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    }

    // Load button
    if (player_type == (int) PlayerType::zwikPlayer) {
      ImGui::TableNextColumn();
      ImGui::InputText("", ai_name, 10);
      ImGui::TableNextColumn();
      if (ImGui::Button("Load Gene", ImVec2(-1.0f, 0.0f))) {
        player_mutex.lock();
        delete game->players[player_id]->agent;
        std::ifstream file("ais/" + (std::string) ai_name);
        std::string ai_gene;
        file >> ai_gene;
        file.close();
        game->players[player_id]->agent = new AIZwikPlayer(game->players[player_id], ai_gene);
        player_mutex.unlock();
      }
      ImGui::TableNextColumn();
      ImGui::Text(
              (std::string("Hash: ")
               + std::to_string(
                       ((NeuralWeb*) game->players[player_id]->agent->get_custom_player_attribute())->get_gene_hash())
               ).c_str());

      ImGui::TableNextRow();
    }

    ImGui::TableNextColumn(); ImGui::Text("Longest Route");

    player_mutex.lock();
    ImGui::TableNextColumn(); ImGui::Text("%i", game->players[player_id]->longest_route);
    player_mutex.unlock();

    ImGui::TableNextColumn(); ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("End Turn").x);

    ImGui::TableNextColumn();
    player_mutex.lock();
    ImGui::Text("%i VP", game->players[player_id]->victory_points);
    player_mutex.unlock();

    ImGui::EndTable();
  }

  player_mutex.lock();
  PlayerType player_type = game->players[player_id]->agent->get_player_type();
  player_mutex.unlock();

  if (player_type == PlayerType::guiPlayer && app_info->state != AppState::Replaying) {
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
                MoveType move_type = game->players[player_id]->available_moves[move_i].type;
                int move_index_id = game->players[player_id]->available_moves[move_i].index;
                player_mutex.unlock();

                if (move_type == MoveType::NoMove && !show_all_moves) {
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
                    moves[player_id].type = index_move(current_structure[player_id]);
                    moves[player_id].index = game->players[player_id]->available_moves[move_i].index;

                    game->human_input_received(moves[player_id]);
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

  if (ImGui::CollapsingHeader("Development Cards")) {
    if (ImGui::BeginTable("split", 1)) {
      ImGui::TableNextColumn(); ImGui::Text("Played: ");
      player_mutex.lock();
      ImGui::TableNextColumn(); ImGui::Text("Knights: %i", game->players[player_id]->played_knight_cards);
      player_mutex.unlock();


      ImGui::TableNextColumn(); ImGui::Text("Available: ");
      player_mutex.lock();
      for (DevelopmentCard const& dev_card : game->players[player_id]->development_cards) {
        ImGui::TableNextColumn(); ImGui::Text("%s", dev_card_names_char[dev_card.type]);
      }
      player_mutex.unlock();

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
        MoveType move_type = game->players[player_id]->available_moves[move_i].type;
        player_mutex.unlock();

        if (move_type == MoveType::NoMove && !show_all_moves) {
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
