#ifndef FASTCATAN_COMPONENTS_H
#define FASTCATAN_COMPONENTS_H

/******************
 *     RULES      *
 ******************/

// Map Layout
static const int amount_of_tiles = 19;
static const int amount_of_corners = 54;
static const int amount_of_streets = 71;

static const int tile_rows = 5;
static const int corner_rows = 6;

static const int corners_per_row[6] = {7, 9, 11, 11, 9, 7};
static const int tiles_in_row[tile_rows] = {3, 4, 5, 4, 3};

static const int amount_of_tokens = 18;
static const int max_number_tokens[11] = {1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1};

// Max amount of tile_array included in the game
static const int max_terrain_tiles[6] = {3, 4, 3, 4, 4, 1};

// Harbors
static const int max_harbors[9] = {4, 1, 1, 1, 1, 1};


/******************
 *     COLORS     *
 ******************/

enum Color {
    Green,
    Red,
    White,
    Blue,
    NoColor
};

inline Color index_color(int color_index) { return (Color) color_index; }
inline int color_index(Color color) { return (int) color; }


/******************
 *     CARDS      *
 ******************/

enum CardType {
    Brick,
    Lumber,
    Ore,
    Grain,
    Wool,
    NoCard
};

inline CardType index_card(int card_index) { return (CardType) card_index; }
inline int card_index(CardType card) { return (int) card; }


/******************
 *     TILES      *
 ******************/

enum TileType {
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture,
    Desert
};

inline TileType index_tile(int tile_index) { return (TileType) tile_index; }
inline int tile_index(TileType tile) { return (int) tile; }

inline CardType tile2card(TileType tile) { return (CardType) tile; }
inline TileType card2tile(CardType card) { return (TileType) card; }


static const char* tile_names[] = {
    "Hills",
    "Forest",
    "Mountains",
    "Fields",
    "Pasture",
    "Desert"
};

static const char tile_shortnames[] = {
    'H',  // Hills
    'F',  // Forest
    'M',  // Mountains
    'f',  // Fields
    'P',  // Pasture
    'D'   // Desert
};


/******************
 *    CORNERS     *
 ******************/

enum CornerOccupancy {
    EmptyCorner,
    Village,
    City
};

static const char corner_shortnames[] = {
    '.',  // EmptyCorner
    'g',  // GreenVillage
    'G',  // GreenCity
    'r',  // RedVillage
    'R',  // RedCity
    'w',  // WhiteVillage
    'W',  // WhiteCity
    'b',  // BlueVillage
    'B'   // BlueCity
};


/******************
 *    HARBORS     *
 ******************/

enum HarborType {
    Harbor_None,
    Harbor_Generic,
    Harbor_Brick,
    Harbor_Grain,
    Harbor_Wool,
    Harbor_Lumber,
    Harbor_Ore
};


/******************
 *     MOVES      *
 ******************/

enum MoveType {
  buildStreet,
  buildVillage,
  buildCity,
  buyDevelopment,
  Trade,
  Exchange,
  openingMove,
  NoMove
};


#endif //FASTCATAN_COMPONENTS_H
