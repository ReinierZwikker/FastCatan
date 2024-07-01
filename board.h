//
// Created by reini on 25/04/2024.
//

#ifndef FASTCATAN_BOARD_H
#define FASTCATAN_BOARD_H

enum tile_type {
  Desert,
  Hills,
  Forest,
  Mountains,
  Fields,
  Pasture
};

static const char* tile_names[] = {"Desert", "Hills", "Forest", "Mountains", "Fields", "Pasture"};

enum corner_occupancy {
  EmptyCorner,
  GreenVillage,
  GreenCity,
  RedVillage,
  RedCity,
  WhiteVillage,
  WhiteCity,
  BlueVillage,
  BlueCity
};

enum street_occupancy {
  EmptyStreet,
  GreenStreet,
  RedStreet,
  WhiteStreet,
  BlueStreet,
};

struct Street {
  street_occupancy occupancy; // What is occupying the corner
};

struct Corner {
  corner_occupancy occupancy; // What is occupying the corner
};

struct Tile {
  int number_token;  // The number on the tile
  tile_type type;  // The type of tile
  bool robber;  // Is a robber occupying the tile

  corner_occupancy *corners[6];  // pointer list of corners, starts counting in the top left corner
  street_occupancy *streets[6];  // pointer list of streets, starts counting at the top left line
};

struct Board {

  Board();

  constexpr static const int amount_of_tiles = 19;

  // Initialize Empty
  Tile tiles[amount_of_tiles]{};
  corner_occupancy corners[54] = {corner_occupancy::EmptyCorner};
  street_occupancy streets[71] = {street_occupancy::EmptyStreet};

  // Map layout (amount of tiles in every row)
  constexpr static const int board_rows = 5;
  constexpr static const int tiles_in_row[board_rows] = {3, 4, 5, 4, 3};
  int tile_diff[board_rows]{};
  int row_decrease[board_rows] = {0};
  int previous_rows[board_rows + 1] = {0};

  // Max amount of tiles included in the game
  constexpr static const int max_terrain_tiles[6] = {1, 3, 4, 3, 4, 4};
  constexpr static const tile_type tile_order[6] = {
    Desert, Hills, Forest, Mountains, Fields, Pasture
  };
};

#endif //FASTCATAN_BOARD_H
