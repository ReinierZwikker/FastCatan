#include "window_board.h"
#include "iostream"

static int tile_id = 0;
static int corner_id = 0;
int current_item;
static TileSelectionItem tile_selection_item{};
static CornerSelectionItem corner_selection_item{};
static bool refresh_map;

void WindowBoard(Game* game, ViewPort* viewport) {
  if (ImGui::Button("Reshuffle")) {
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
    if (current_number != game->board.tile_array[tile_id].number_token) {
      game->board.tile_array[tile_id].number_token = current_number;
      refresh_map = true;
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
      refresh_map = true;
    }
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
    current_item = game->board.corner_array[corner_id].occupancy;
    ImGui::Combo("Corner Occupancy", &current_item, "EmpyCorner\0Village\0City\0\0");
    if (current_item != game->board.corner_array[corner_id].occupancy) {
      game->board.corner_array[corner_id].occupancy = static_cast<CornerOccupancy>(current_item);
      refresh_map = true;
    }

    // Corner Color
    current_item = game->board.corner_array[corner_id].color;
    ImGui::Combo("Color", &current_item, "Green\0Red\0White\0Blue\0NoColor\0\0");
    if (current_item != game->board.corner_array[corner_id].color) {
      game->board.corner_array[corner_id].color = static_cast<Color>(current_item);
      refresh_map = true;
    }
  }

  if (refresh_map) {
    viewport->NewMap(game);
    refresh_map = false;
  }
}
