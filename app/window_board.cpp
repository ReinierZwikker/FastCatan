#include "window_board.h"
#include "iostream"
#include <mutex>

std::mutex board_mutex;

static int tile_id = 0;
static int corner_id = 0;
static int street_id = 0;

int current_item;
int current_corner;
int current_street;

static SelectionItem tile_selection_item{};
static SelectionItem corner_selection_item{};
static SelectionItem street_selection_item{};
static bool refresh_map;

void WindowBoard(Game* game, ViewPort* viewport) {
  if (ImGui::Button("Reshuffle")) {
    game->board.Randomize();
    refresh_map = true;
  }

  if (ImGui::CollapsingHeader("Tiles")) {
    ImGui::InputInt("Tile ID", &tile_id);
    if (tile_id < 0) {
      tile_id = 0;
    }
    else if (tile_id > amount_of_tiles - 1) {
      tile_id = amount_of_tiles - 1;
    }
    tile_selection_item.id = tile_id;
    tile_selection_item.game = game;
    tile_selection_item.render = true;
    viewport->tile_selection_item = tile_selection_item;

    // Number token
    int current_number = game->board.tile_array[tile_id].number_token;
    ImGui::InputInt("Number Token", &current_number);
    if (current_number > 12) {
      current_number = 12;
    }
    else if (current_number < 2) {
      current_number = 2;
    }
    if (current_number != game->board.tile_array[tile_id].number_token) {
      game->board.tile_array[tile_id].number_token = current_number;
    }

    // Tile Type
    current_item = game->board.tile_array[tile_id].type;
    ImGui::Combo("Tile Type", &current_item, "Hills\0Forest\0Mountains\0Fields\0Pasture\0Desert\0\0");
    if (current_item != game->board.tile_array[tile_id].type) {
      game->board.tile_array[tile_id].type = static_cast<TileType>(current_item);
      refresh_map = true;
    }

    // Robber
    current_item = game->board.tile_array[tile_id].robber;
    ImGui::Combo("Robber", &current_item, "Empty\0Robber\0\0");
    if (current_item != game->board.tile_array[tile_id].robber) {
      game->board.tile_array[tile_id].robber = current_item;
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    // Corners
    ImGui::InputInt("Corner", &current_corner);
    if (current_corner < 0) {
      current_corner = 5;
    }
    else if (current_corner > 5) {
      current_corner = 0;
    }
    corner_id = game->board.tile_array[tile_id].corners[current_corner]->id;

    // Streets
    ImGui::InputInt("Street##0", &current_street);
    if (current_street < 0) {
      current_street = 5;
    }
    else if (current_street > 5) {
      current_street = 0;
    }
    street_id = game->board.tile_array[tile_id].streets[current_street]->id;

  }

  if (ImGui::CollapsingHeader("Corners")) {
    ImGui::InputInt("Corner ID", &corner_id);
    if (corner_id < 0) {
      corner_id = 0;
    }
    else if (corner_id > amount_of_corners - 1) {
      corner_id = amount_of_corners - 1;
    }
    corner_selection_item.id = corner_id;
    corner_selection_item.game = game;
    corner_selection_item.render = true;
    viewport->corner_selection_item = corner_selection_item;

    // Corner Occupancy
    board_mutex.lock();
    current_item = game->board.corner_array[corner_id].occupancy;
    ImGui::Combo("Corner Occupancy", &current_item, "EmptyCorner\0Village\0City\0\0");
    if (current_item != game->board.corner_array[corner_id].occupancy) {
      game->board.corner_array[corner_id].occupancy = static_cast<CornerOccupancy>(current_item);
    }
    board_mutex.unlock();

    // Corner Color
    board_mutex.lock();
    current_item = game->board.corner_array[corner_id].color;
    ImGui::Combo("Corner Color", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
    if (current_item != game->board.corner_array[corner_id].color) {
      game->board.corner_array[corner_id].color = static_cast<Color>(current_item);
    }
    board_mutex.unlock();

    // Corner Harbor
    board_mutex.lock();
    current_item = game->board.corner_array[corner_id].harbor;
    ImGui::Combo("Corner Harbor", &current_item, "None\0Generic\0Brick\0Grain\0Wool\0Lumber\0Ore\0\0");
    if (current_item != game->board.corner_array[corner_id].harbor) {
      game->board.corner_array[corner_id].harbor = static_cast<HarborType>(current_item);
    }
    board_mutex.unlock();

    // Streets
    if (ImGui::TreeNode("Streets##2")) {
      board_mutex.lock();
      Street* street_0 = game->board.corner_array[corner_id].streets[0];
      Street* street_1 = game->board.corner_array[corner_id].streets[1];
      Street* street_2 = game->board.corner_array[corner_id].streets[2];

      if (street_0 != nullptr) {
        current_item = game->board.street_array[street_0->id].color;
        ImGui::Text("Street left [%i]", street_0->id);
        ImGui::Combo("Street Color##1", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
        if (current_item != game->board.street_array[street_0->id].color) {
          game->board.street_array[street_0->id].color = static_cast<Color>(current_item);
        }
      }
      else {
        ImGui::Text("Street left - Not connected");
      }

      if (street_1 != nullptr) {
        current_item = game->board.street_array[street_1->id].color;
        ImGui::Text("Street above/below [%i]", street_1->id);
        ImGui::Combo("Street Color##2", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
        if (current_item != game->board.street_array[street_1->id].color) {
          game->board.street_array[street_1->id].color = static_cast<Color>(current_item);
        }
      }
      else {
        ImGui::Text("Street above/below - Not connected");
      }

      if (street_2 != nullptr) {
        current_item = game->board.street_array[street_2->id].color;
        ImGui::Text("Street right [%i]", street_2->id);
        ImGui::Combo("Street Color##3", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
        if (current_item != game->board.street_array[street_2->id].color) {
          game->board.street_array[street_2->id].color = static_cast<Color>(current_item);
        }
      }
      else {
        ImGui::Text("Street right - Not connected");
      }

      board_mutex.unlock();
      ImGui::TreePop();
    }
  }

  if (ImGui::CollapsingHeader("Streets")) {
    ImGui::InputInt("Street ID", &street_id);
    if (street_id < 0) {
      street_id = 0;
    }
    else if (street_id > amount_of_streets - 1) {
      street_id = amount_of_streets - 1;
    }
    street_selection_item.id = street_id;
    street_selection_item.game = game;
    street_selection_item.render = true;
    viewport->street_selection_item = street_selection_item;

    // Street Color
    board_mutex.lock();
    current_item = game->board.street_array[street_id].color;
    ImGui::Combo("Street Color", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
    if (current_item != game->board.street_array[street_id].color) {
      game->board.street_array[street_id].color = static_cast<Color>(current_item);
    }
    board_mutex.unlock();

    if (ImGui::TreeNode("Corners##2")) {
      board_mutex.lock();
      Corner* corner_0 = game->board.street_array[street_id].corners[0];
      Corner* corner_1 = game->board.street_array[street_id].corners[1];

      if (corner_0 != nullptr) {
        current_item = game->board.corner_array[corner_0->id].color;
        ImGui::Text("Corner 1 [%i]", corner_0->id);
        ImGui::Combo("Corner Color##1", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
        if (current_item != game->board.corner_array[corner_0->id].color) {
          game->board.corner_array[corner_0->id].color = static_cast<Color>(current_item);
        }
      }
      else {
        ImGui::Text("Corner 1 - Not connected");
      }

      if (corner_1 != nullptr) {
        current_item = game->board.corner_array[corner_1->id].color;
        ImGui::Text("Corner 2 [%i]", corner_1->id);
        ImGui::Combo("Corner Color##2", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
        if (current_item != game->board.corner_array[corner_1->id].color) {
          game->board.corner_array[corner_1->id].color = static_cast<Color>(current_item);
        }
      }
      else {
        ImGui::Text("Corner 2- Not connected");
      }

      board_mutex.unlock();
      ImGui::TreePop();
    }
  }

  if (refresh_map) {
    viewport->NewMap(game);
    refresh_map = false;
  }
}
