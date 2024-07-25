#ifndef FASTCATAN_BOARD_H
#define FASTCATAN_BOARD_H

#include "components.h"

struct Harbor {
  int tile_id;
  int corner_1;
  int corner_2;
  HarborType type;
};

struct Street {
  Color color = Color::NoColor;  // What is occupying the corner
};

struct Corner {
  CornerOccupancy occupancy = CornerOccupancy::EmptyCorner;  // What is occupying the corner
  Color color = Color::NoColor;
  HarborType harbor = HarborType::Harbor_None;  // What type of harbor is on this corner

  Street *streets[3] = {nullptr, nullptr, nullptr}; // pointer list of streets that are connected to this corner, starting at the vertical street
};

struct Tile {
  int number_token;  // The number on the tile
  TileType type;  // The type of tile
  bool robber;  // Is a robber occupying the tile

  Corner *corners[6];  // pointer list of corners, starts counting in the top left corner
  Street *streets[6];  // pointer list of streets, starts counting at the top left line
};

struct Board {
public:
  Board();

  // Initialize Empty
  Tile tile_array[amount_of_tiles]{};
  Tile *tiles[tile_rows]{};

  Corner corner_array[amount_of_corners];
  Corner *corners[corner_rows]{};
  Street street_array[amount_of_streets]{};
  Street *streets[street_rows]{};

  void PrintBoard();



private:
  void LinkParts();

  void InitializeTilesAndTokens();

  void ShuffleTilesAndTokens();
  void AddNumberTokensToTiles();
  bool CheckNumberTokens();

  void RewriteBoardLayout();
  void LinkCornersAndStreetsToTiles();
  void LinkStreetsToCorners();

  void AddHarbors();


  int tile_diff[tile_rows]{};
  int row_decrease[tile_rows] = {0};
  int previous_rows[tile_rows + 1] = {0};

  // Max amount of number tokens in the game
  /* From the rule book:
   * The 18 number tokens are marked with the numerals "2" through "12".
   * There is only one "2" and one "12". There is no "7".*/
  int number_tokens[amount_of_tokens]{};

  bool show_number_token_debug = false;

  // Hardcoded for now:
  Harbor harbor_1 = Harbor(0, 0, 1, HarborType::Harbor_Generic);
  Harbor harbor_2 = Harbor(1, 1, 2, HarborType::Harbor_Grain);
  Harbor harbor_3 = Harbor(6, 1, 2, HarborType::Harbor_Ore);
  Harbor harbor_4 = Harbor(11, 2, 3, HarborType::Harbor_Generic);
  Harbor harbor_5 = Harbor(15, 3, 4, HarborType::Harbor_Wool);
  Harbor harbor_6 = Harbor(17, 3, 4, HarborType::Harbor_Generic);
  Harbor harbor_7 = Harbor(16, 4, 5, HarborType::Harbor_Generic);
  Harbor harbor_8 = Harbor(12, 5, 0, HarborType::Harbor_Brick);
  Harbor harbor_9 = Harbor(3, 5, 0, HarborType::Harbor_Lumber);

  const Harbor harbors[9] = {harbor_1, harbor_2, harbor_3, harbor_4, harbor_5, harbor_6, harbor_7, harbor_8, harbor_9};

};

#endif //FASTCATAN_BOARD_H
