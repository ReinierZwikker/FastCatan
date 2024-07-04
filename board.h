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

static const char tile_shortnames[] = {
        'D',  // Desert
        'H',  // Hills
        'F',  // Forest
        'M',  // Mountains
        'f',  // Fields
        'P'   // Pasture
};

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
  BlueStreet
};

enum harbor_types {
  None,
  Generic,
  Brick,
  Grain,
  Wool,
  Lumber,
  Ore
};

struct Harbor {
  int tile_id;
  int corner_1;
  int corner_2;
  harbor_types type;
};

struct Street {
  street_occupancy occupancy = street_occupancy::EmptyStreet;  // What is occupying the corner
};

struct Corner {
  corner_occupancy occupancy = corner_occupancy::EmptyCorner;  // What is occupying the corner
  harbor_types harbor = harbor_types::None;  // What type of harbor is on this corner
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

  // Initialize Empty
  Tile tiles[amount_of_tiles]{};
  Corner corners[54]{};
  Street streets[71]{};

  void print_board();

private:
  void CalculateTileDifference();
  void InitializeTilesAndTokens();

  void ShuffleTilesAndTokens();
  void AddNumberTokensToTiles();
  bool CheckNumberTokens();

  void RewriteBoardLayout();
  void LinkCornersAndStreetsToTiles();

  void AddHarbors();

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
  Harbor harbor_1 = Harbor(0, 0, 1, harbor_types::Generic);
  Harbor harbor_2 = Harbor(1, 1, 2, harbor_types::Grain);
  Harbor harbor_3 = Harbor(6, 1, 2, harbor_types::Ore);
  Harbor harbor_4 = Harbor(11, 2, 3, harbor_types::Generic);
  Harbor harbor_5 = Harbor(15, 3, 4, harbor_types::Wool);
  Harbor harbor_6 = Harbor(17, 3, 4, harbor_types::Generic);
  Harbor harbor_7 = Harbor(16, 4, 5, harbor_types::Generic);
  Harbor harbor_8 = Harbor(12, 5, 0, harbor_types::Brick);
  Harbor harbor_9 = Harbor(3, 5, 0, harbor_types::Lumber);

  const Harbor harbors[9] = {harbor_1, harbor_2, harbor_3, harbor_4, harbor_5, harbor_6, harbor_7, harbor_8, harbor_9};

};

#endif //FASTCATAN_BOARD_H
