//
// Created by reini on 25/04/2024.
//

#include <random>
#include <stdexcept>

#include "board.h"

Board::Board() {



  int current_tile_i = 0;
  for (int tile_type_i = 0; tile_type_i < 6; tile_type_i++) {
    for (int tile_i = 0; tile_i < max_terrain_tiles[tile_type_i]; tile_i++) {
      tiles[current_tile_i] = tile_order[tile_type_i];
      current_tile_i++;
    }
  }
  if (current_tile_i != 18) { throw std::invalid_argument("Tiles do not add up!"); }
  // Shuffle tiles
}
