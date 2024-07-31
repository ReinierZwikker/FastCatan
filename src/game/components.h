#ifndef FASTCATAN_COMPONENTS_H
#define FASTCATAN_COMPONENTS_H

#include <string>

struct Corner;
struct Street;

/******************
 *     RULES      *
 ******************/

// Map Layout
static const int amount_of_tiles = 19;
static const int amount_of_corners = 54;
static const int amount_of_streets = 72;

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

static const int max_available_moves = 200;
static const int moves_per_turn = 25;


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

static const std::string color_names[] = {
    "Green",
    "Red",
    "White",
    "Blue"
};

static const std::string color_offsets[] = {
    "",
    "  ",
    "",
    " "
};

inline Color index_color(int color_index) { return (Color) color_index; }
inline int color_index(Color color) { return (int) color; }

inline std::string color_name(Color color) { return color_names[color_index(color)]; }
inline std::string color_offset(Color color) { return color_offsets[color_index(color)]; }

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

static const std::string card_names[] = {
    "Brick",
    "Lumber",
    "Ore",
    "Grain",
    "Wool"
};

inline CardType index_card(int card_index) { return (CardType) card_index; }
inline int card_index(CardType card) { return (int) card; }

inline std::string card_name(CardType card) { return card_names[card_index(card)]; }

/******************
 *    Street     *
 ******************/

struct Street {
  int id = -1;
  Color color = Color::NoColor;  // Who is occupying the street
  Corner *corner_1 = nullptr;
  Corner *corner_2 = nullptr;
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
  int id = -1;
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
  buildStreet,    // Specify: Street Index
  buildVillage,   // Specify: Corner Index
  buildCity,      // Specify: Corner Index
  buyDevelopment, // Specify: -
  Trade,          // Specify: Other Player, Transmitting Card, Receiving Card, Amount
  Exchange,       // Specify: Transmitting Card, Receiving Card, Amount due to Harbor
  moveRobber,     // Specify: Tile Index
  endTurn,
  NoMove
};

inline MoveType index_move(int move_index) { return (MoveType) move_index; }
inline int move_index(MoveType move) { return (int) move; }

enum TurnType {
  openingTurnVillage,
  openingTurnStreet,
  normalTurn,
  robberTurn,
  tradeTurn,
  noTurn
};

struct Move {
  inline Move() {
    move_type = NoMove;
    index = -1;
    other_player = NoColor;
    tx_card = NoCard;
    rx_card = NoCard;
    tx_amount = -1;
    rx_amount = -1;
  }
  // Move template, only set applicable fields when communicating moves
  MoveType move_type;
  int index;
  Color other_player;
  CardType tx_card;
  CardType rx_card;
  int tx_amount;
  int rx_amount;
};


inline bool operator==(const Move& move_lhs, const Move& move_rhs) {
  if (move_lhs.move_type    == move_rhs.move_type    &&
      move_lhs.index        == move_rhs.index        &&
      move_lhs.other_player == move_rhs.other_player &&
      move_lhs.tx_card      == move_rhs.tx_card      &&
      move_lhs.rx_card      == move_rhs.rx_card      &&
      move_lhs.tx_amount    == move_rhs.tx_amount    &&
      move_lhs.rx_amount    == move_rhs.rx_amount)
  { return true; } else { return false; }
}

inline std::string move2string(Move move) {
  switch (move.move_type) {
    case buildStreet:
      return "Building Street at street " + std::to_string(move.index);
    case buildVillage:
      return "Building Village at corner " + std::to_string(move.index);
    case buildCity:
      return "Building City at corner " + std::to_string(move.index);
    case buyDevelopment:
      return "Buying one Development Card";
    case Trade:
      return "Trading "
        + std::to_string(move.tx_amount) + " " + card_name(move.tx_card)
        + " for "
        + std::to_string(move.rx_amount) + " " + card_name(move.rx_card);
    case Exchange:
      return "Exchanging "
             + std::to_string(move.tx_amount) + " " + card_name(move.tx_card)
             + " for "
             + std::to_string(move.rx_amount) + " " + card_name(move.rx_card)
             + "with the bank";
    case moveRobber:
      return "Moving Robber to tile " + std::to_string(move.index);
    case endTurn:
      return "Ending my turn!";
    case NoMove:
      return "Invalid Move";
  }
  return "Invalid Move";
}


#endif //FASTCATAN_COMPONENTS_H
