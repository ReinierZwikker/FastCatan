#ifndef FASTCATAN_BOARD_H
#define FASTCATAN_BOARD_H

#include "components.h"

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

  int tile_diff[tile_rows]{};
  int row_decrease[tile_rows] = {0};
  int previous_rows[tile_rows + 1] = {0};

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

  // Max amount of number tokens in the game
  /* From the rule book:
   * The 18 number tokens are marked with the numerals "2" through "12".
   * There is only one "2" and one "12". There is no "7".*/
  int number_tokens[amount_of_tokens]{};

  bool show_number_token_debug = false;

  // Hardcoded for now:
//  Harbor harbor_1 = Harbor(0, 0, 1, HarborType::Harbor_Generic);
//  Harbor harbor_2 = Harbor(1, 1, 2, HarborType::Harbor_Grain);
//  Harbor harbor_3 = Harbor(6, 1, 2, HarborType::Harbor_Ore);
//  Harbor harbor_4 = Harbor(11, 2, 3, HarborType::Harbor_Generic);
//  Harbor harbor_5 = Harbor(15, 3, 4, HarborType::Harbor_Wool);
//  Harbor harbor_6 = Harbor(17, 3, 4, HarborType::Harbor_Generic);
//  Harbor harbor_7 = Harbor(16, 4, 5, HarborType::Harbor_Generic);
//  Harbor harbor_8 = Harbor(12, 5, 0, HarborType::Harbor_Brick);
//  Harbor harbor_9 = Harbor(3, 5, 0, HarborType::Harbor_Lumber);
//
//  const Harbor harbors[9] = {harbor_1, harbor_2, harbor_3, harbor_4, harbor_5, harbor_6, harbor_7, harbor_8, harbor_9};

};

#endif //FASTCATAN_BOARD_H
