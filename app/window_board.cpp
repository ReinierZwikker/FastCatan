#include "window_board.h"
#include "iostream"

static int tile_id = 0;
int current_item;
static TileSelectionItem tile_selection_item{};

void WindowBoard(Game* game, ViewPort* viewport) {
  if (ImGui::Button("Reshuffle")) {
    viewport->NewMap(game);
  }

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
    viewport->NewMap(game);
  }

  // Tile Type
  current_item = game->board.tile_array[tile_id].type;
  ImGui::Combo("Tile Type", &current_item, "Hills\0Forest\0Mountains\0Fields\0Pasture\0Desert\0\0");
  if (current_item != game->board.tile_array[tile_id].type) {
    game->board.tile_array[tile_id].type = static_cast<TileType>(current_item);
    viewport->NewMap(game);
  }

  current_item = game->board.tile_array[tile_id].robber;
  ImGui::Combo("Robber", &current_item, "Empty\0Robber\0\0");
  if (current_item != game->board.tile_array[tile_id].robber) {
    game->board.tile_array[tile_id].robber = current_item;
    viewport->NewMap(game);
  }

}
