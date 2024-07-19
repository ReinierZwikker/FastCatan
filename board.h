#ifndef FASTCATAN_BOARD_H
#define FASTCATAN_BOARD_H

#include "components.h"

struct Harbor {
  int tile_id;
  int corner_1;
  int corner_2;
  harbor_types type;
};

struct Street {
  colors color = colors::Color_None;  // What is occupying the corner
};

struct Corner {
  corner_occupancy occupancy = corner_occupancy::EmptyCorner;  // What is occupying the corner
  colors color = colors::Color_None;
  harbor_types harbor = harbor_types::Harbor_None;  // What type of harbor is on this corner

  Street *streets[3] = {nullptr, nullptr, nullptr}; // pointer list of streets that are connected to this corner, starting at the vertical street
};

struct Tile {
  int number_token;  // The number on the tile
  tile_type type;  // The type of tile
  bool robber;  // Is a robber occupying the tile

  Corner *corners[6];  // pointer list of corners, starts counting in the top left corner
  Street *streets[6];  // pointer list of streets, starts counting at the top left line
};

struct Board {
public:
  Board();

  constexpr static const int amount_of_tiles = 19;
  constexpr static const int amount_of_tokens = 18;
  constexpr static const int corners_per_row[6] = {7, 9, 11, 11, 9, 7};


  // Initialize Empty
  Tile tile_array[amount_of_tiles]{};
  Tile *tiles[6]{};

  Corner corner_array[54];
  Corner *corners[6]{};
  Street streets[71]{};

  void PrintBoard();

  bool CheckValidity();

private:
  void CalculateTileDifference();
  void InitializeTilesAndTokens();

  void ShuffleTilesAndTokens();
  void AddNumberTokensToTiles();
  bool CheckNumberTokens();

  void RewriteBoardLayout();
  void LinkCornersAndStreetsToTiles();
  void LinkStreetsToCorners();

  void AddHarbors();

  // Map layout (amount of tile_array in every row)
  constexpr static const int board_rows = 5;
  constexpr static const int tiles_in_row[board_rows] = {3, 4, 5, 4, 3};
  int tile_diff[board_rows]{};
  int row_decrease[board_rows] = {0};
  int previous_rows[board_rows + 1] = {0};

  // Max amount of tile_array included in the game
  constexpr static const int max_terrain_tiles[6] = {1, 3, 4, 3, 4, 4};
  constexpr static const tile_type tile_order[6] = {
    Desert, Hills, Forest, Mountains, Fields, Pasture
  };

  // Max amount of number tokens in the game
  /* From the rule book:
   * The 18 number tokens are marked with the numerals "2" through "12".
   * There is only one "2" and one "12". There is no "7".*/
  constexpr static const int max_number_tokens[11] = {1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1};
  int number_tokens[amount_of_tokens]{};

  bool show_number_token_debug = false;

  // Harbors
  constexpr static const int max_harbors[9] = {4, 1, 1, 1, 1, 1};

  // Hardcoded for now:
  Harbor harbor_1 = Harbor(0, 0, 1, harbor_types::Harbor_Generic);
  Harbor harbor_2 = Harbor(1, 1, 2, harbor_types::Harbor_Grain);
  Harbor harbor_3 = Harbor(6, 1, 2, harbor_types::Harbor_Ore);
  Harbor harbor_4 = Harbor(11, 2, 3, harbor_types::Harbor_Generic);
  Harbor harbor_5 = Harbor(15, 3, 4, harbor_types::Harbor_Wool);
  Harbor harbor_6 = Harbor(17, 3, 4, harbor_types::Harbor_Generic);
  Harbor harbor_7 = Harbor(16, 4, 5, harbor_types::Harbor_Generic);
  Harbor harbor_8 = Harbor(12, 5, 0, harbor_types::Harbor_Brick);
  Harbor harbor_9 = Harbor(3, 5, 0, harbor_types::Harbor_Lumber);

  const Harbor harbors[9] = {harbor_1, harbor_2, harbor_3, harbor_4, harbor_5, harbor_6, harbor_7, harbor_8, harbor_9};


};

#endif //FASTCATAN_BOARD_H
