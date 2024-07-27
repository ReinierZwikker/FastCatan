#ifndef FASTCATAN_COMPONENTS_H
#define FASTCATAN_COMPONENTS_H

struct Corner;
struct Street;

/******************
 *     RULES      *
 ******************/

// Map Layout
static const int amount_of_tiles = 19;
static const int amount_of_corners = 54;
static const int amount_of_streets = 71;

static const int tile_rows = 5;
static const int corner_rows = 6;
static const int street_rows = 11;

static const int tiles_in_row[tile_rows] = {3, 4, 5, 4, 3};
static const int corners_in_row[corner_rows] = {7, 9, 11, 11, 9, 7};
static const int streets_in_row[street_rows] = {6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6};

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
 *    Street     *
 ******************/

struct Street {
  Color color;  // What is occupying the street
  Corner *corner_1;
  Corner *corner_2;
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

struct Harbor {
  int tile_id;
  int corner_1;
  int corner_2;
  HarborType type;
};

/******************
 *    CORNERS     *
 ******************/

enum CornerOccupancy {
    EmptyCorner,
    Village,
    City
};

struct Corner {
  CornerOccupancy occupancy = CornerOccupancy::EmptyCorner;  // What is occupying the corner
  Color color = Color::NoColor;
  HarborType harbor = HarborType::Harbor_None;  // What type of harbor is on this corner

  Street *streets[3] = {nullptr, nullptr, nullptr}; // pointer list of streets that are connected to this corner, starting at the vertical street
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

struct Tile {
  int number_token;  // The number on the tile
  TileType type;  // The type of tile
  float color[3];  // Color when displayed in OpenGL
  bool robber;  // Is a robber occupying the tile

  Corner *corners[6];  // pointer list of corners, starts counting in the top left corner
  Street *streets[6];  // pointer list of streets, starts counting at the top left line
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

struct Move {
  // Move template, only set applicable fields when communicating moves
  MoveType move_type = NoMove;
  int index = -1;
  CornerOccupancy *corner = nullptr;
  CornerOccupancy *street = nullptr;
  Color other_player = NoColor;
  CardType rx_card = NoCard;
  CardType tx_card = NoCard;
  int amount = -1;
};


#endif //FASTCATAN_COMPONENTS_H
