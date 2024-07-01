//
// Created by reini on 25/04/2024.
//

#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>

#include "board.h"

Board::Board() {

  // ## Initialize tiles ##
  int current_tile_i = 0;
  for (int tile_type_i = 0; tile_type_i < 6; tile_type_i++) {
    for (int tile_i = 0; tile_i < max_terrain_tiles[tile_type_i]; tile_i++) {
      // Construct a new tile
      Tile current_tile = {};

      // Set initial values
      current_tile.type = tile_order[tile_type_i];
      current_tile.robber = false;

      // Append to tiles
      tiles[current_tile_i] = current_tile;
      current_tile_i++;
    }
  }
  if (current_tile_i != amount_of_tiles) { throw std::invalid_argument("Tiles do not add up!"); }

  // ## Shuffle tiles ##
  auto random_seed = std::random_device {};
  auto rng = std::default_random_engine {random_seed()};
  std::shuffle(tiles, tiles+amount_of_tiles, rng);

  // ## Rewrite the map layout form ##
  for (int row = 0; row < board_rows; row++) {
    if (row < board_rows - 1){
      tile_diff[row] = tiles_in_row[row] - tiles_in_row[row + 1];

      if (tile_diff[row] == 1) {
        row_decrease[row + 1] = 1;
      }
      else if (tile_diff[row] == -1) {
        row_decrease[row + 1] = 0;
      }
      else {
        throw std::invalid_argument("Map layout not supported!");
      }
    }
    else {
      row_decrease[row + 1] = 1;
    }

    previous_rows[row + 1] = previous_rows[row] + 2 * tiles_in_row[row] + 1 + 2 * row_decrease[row];
  }
  previous_rows[0] = 0;  // I don't know why this has to be there, but it breaks without it

  // ## Link tiles ##
  int current_column = 0;
  int current_row = 0;

  for (int tile_i = 0; tile_i < amount_of_tiles; tile_i++) {
    // Check if the next row is reached
    if (current_column + 1 > tiles_in_row[current_row]) {
      current_column = 0;
      current_row++;
    }

    // Top side of the tile
    for (int corner_i = 0; corner_i < 3; corner_i++) {
      int corner_id = corner_i + 2 * current_column + previous_rows[current_row] + row_decrease[current_row];
      tiles[tile_i].corners[corner_i] = &corners[corner_id];
      tiles[tile_i].streets[corner_i] = &streets[corner_id];
    }
    // Bottom side of the tile
    for (int corner_i = 0; corner_i < 3; corner_i++) {
      int corner_id = 3 - corner_i + 2 * current_column + previous_rows[current_row + 1] - row_decrease[current_row + 1];
      tiles[tile_i].corners[corner_i + 3] = &corners[corner_id];
      tiles[tile_i].streets[corner_i + 3] = &streets[corner_id];
    }

    current_column++;
  }

//  for (auto & tile : tiles) {
//    std::cout << "Tile type: " << tile_names[tile.type] << std::endl;
//  }
}
