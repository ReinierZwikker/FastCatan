#ifndef FASTCATAN_COMPONENTS_H
#define FASTCATAN_COMPONENTS_H

#include <string>
#include <cstdint>

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

// Max amount of development cards in the game
// {Knight, VP, Monopoly, Research, Streets}
static const int max_development_cards[5] = {14, 5, 2, 2, 2};
static const int amount_of_development_cards = 25;

// Harbors
static const int max_harbors[9] = {4, 1, 1, 1, 1, 1};

static const int max_available_moves = 200;
static const int moves_per_turn = 25;
static const int max_rounds = 500;

/******************
 *     COLORS     *
 ******************/

#pragma pack(push, 1)
enum class Color : uint8_t {
    Green,
    Red,
    White,
    Blue,
    NoColor
};
#pragma pack(pop)

static const std::string color_names[] = {
    "Green",
    "Red",
    "White",
    "Blue",
    "No Color"
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


#pragma pack(push, 1)
enum class CardType : uint8_t {
    Brick,
    Lumber,
    Ore,
    Grain,
    Wool,
    NoCard
};
#pragma pack(pop)

static const std::string card_names[] = {
    "Brick",
    "Lumber",
    "Ore",
    "Grain",
    "Wool",
    "No Card"
};

static const char* card_names_char[] = {
    "Brick",
    "Lumber",
    "Ore",
    "Grain",
    "Wool",
    "No Card"
};

inline CardType index_card(int card_index) { return (CardType) card_index; }
inline int card_index(CardType card) { return (int) card; }

inline std::string card_name(CardType card) { return card_names[card_index(card)]; }

/******************************
 *     Development Cards      *
 ******************************/

enum DevelopmentType {
  Knight,
  VictoryPoint,
  Monopoly,
  YearOfPlenty,
  RoadBuilding,
  None
};

static const std::string dev_card_names[] = {
    "Knight",
    "Victory Point",
    "Monopoly",
    "Year of Plenty",
    "Road Building"
};

static const char* dev_card_names_char[] = {
    "Knight",
    "Victory Point",
    "Monopoly",
    "Year of Plenty",
    "Road Building"
};

struct DevelopmentCard {
  DevelopmentType type = None;
  bool bought_this_round = true;
};

inline DevelopmentType index_dev_card(int dev_card_index) { return (DevelopmentType) dev_card_index; }
inline int dev_card_index(DevelopmentType dev_card) { return (int) dev_card; }

/*****************
 *    Street     *
 *****************/

struct Street {
  int id = -1;
  Color color = Color::NoColor;  // Who is occupying the street
  Corner *corners[2] = {nullptr, nullptr};
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
  int corners[2];
  HarborType type;

  Harbor(int, int, int, HarborType);
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
  float coordinates[2] = {0.0f, 0.0f}; // used by openGL
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
  bool robber;  // Is a robber occupying the tile

  Corner *corners[6];  // pointer list of corners, starts counting in the top left corner
  Street *streets[6];  // pointer list of streets, starts counting at the top left line

  float color[3];  // Color when displayed in OpenGL
  float coordinates[2] = {0.0f, 0.0f};  // used by openGL
};

/******************
 *     MOVES      *
 ******************/

#pragma pack(push, 1)
enum class MoveType : uint8_t {
  buildStreet,     // Specify: Street Index
  buildVillage,    // Specify: Corner Index
  buildCity,       // Specify: Corner Index
  buyDevelopment,  // Specify: -
  playDevelopment, // Specify: Development Index
  Trade,           // Specify: Other Player, Transmitting Card, Receiving Card, Amount
  Exchange,        // Specify: Transmitting Card, Receiving Card, Amount due to Harbor
  moveRobber,      // Specify: Tile Index
  getCardBank,     // Specify: Card Index
  endTurn,
  NoMove,
  Replay
};
#pragma pack(pop)

inline MoveType index_move(int move_index) { return (MoveType) move_index; }
inline int move_index(MoveType move) { return (int) move; }

enum class TurnType : uint8_t {
  openingTurnVillage,
  openingTurnStreet,
  normalTurn,
  robberTurn,
  tradeTurn,
  devTurnKnight,
  devTurnMonopoly,
  devTurnYearOfPlenty,
  devTurnStreet,
  noTurn
};

#pragma pack(push, 1)
struct Move {
  inline Move() {
    type = MoveType::NoMove;
    index = 0;
    other_player = Color::NoColor;
    tx_card = CardType::NoCard;
    rx_card = CardType::NoCard;
    tx_amount = 0;
    rx_amount = 0;
  }
  // Move template, only set applicable fields when communicating moves
  MoveType type = MoveType::NoMove;
  uint16_t index = 0;
  Color other_player = Color::NoColor;
  CardType tx_card = CardType::NoCard;
  CardType rx_card = CardType::NoCard;
  uint8_t tx_amount = 0;
  uint8_t rx_amount = 0;
};
#pragma pack(pop)


inline bool operator==(const Move& move_lhs, const Move& move_rhs) {
  if (move_lhs.type == move_rhs.type &&
      move_lhs.index        == move_rhs.index &&
      move_lhs.other_player == move_rhs.other_player &&
      move_lhs.tx_card      == move_rhs.tx_card &&
      move_lhs.rx_card      == move_rhs.rx_card &&
      move_lhs.tx_amount    == move_rhs.tx_amount &&
      move_lhs.rx_amount    == move_rhs.rx_amount)
  { return true; } else { return false; }
}

inline std::string move2string(Move move) {
  switch (move.type) {
    case MoveType::buildStreet:
      return "Building Street at street " + std::to_string(move.index);
    case MoveType::buildVillage:
      return "Building Village at corner " + std::to_string(move.index);
    case MoveType::buildCity:
      return "Building City at corner " + std::to_string(move.index);
    case MoveType::buyDevelopment:
      return "Buying one Development Card";
    case MoveType::playDevelopment:
      return "Playing the " + dev_card_names[move.index] + " Development Card";
    case MoveType::Trade:
      return "Trading "
        + std::to_string(move.tx_amount) + " " + card_name(move.tx_card)
        + " for "
        + std::to_string(move.rx_amount) + " " + card_name(move.rx_card);
    case MoveType::Exchange:
      return "Exchanging "
             + std::to_string(move.tx_amount) + " " + card_name(move.tx_card)
             + " for "
             + std::to_string(move.rx_amount) + " " + card_name(move.rx_card)
             + " with the bank";
    case MoveType::moveRobber:
      return "Moving Robber to tile " + std::to_string(move.index);
    case MoveType::endTurn:
      return "Ending my turn!";
    case MoveType::NoMove:
      return "Invalid Move";
    case MoveType::getCardBank:
      return "Getting card from the bank";
    case MoveType::Replay:
      return "Replaying";
  }
  return "Invalid Move";
}

/******************
 *    PLAYERS     *
 ******************/

enum class PlayerType : uint8_t {
    consolePlayer,
    guiPlayer,
    randomPlayer,
    zwikPlayer,
    beanPlayer,
    NoPlayer
};

static const char* player_type_char[] = {
    "Console Player",
    "GUI Player",
    "Random Player",
    "Zwik Player",
    "Bean Player",
    "No Player"
};

enum PlayerState {
  Waiting,
  Playing,
  Finished
};

static const char* player_state_char[] = {
    "Waiting",
    "Playing",
    "Thanks for playing!"
};

/******************
 *    Game     *
 ******************/

enum GameStates {
  UnInitialized,
  ReadyToStart,
  SetupRound,
  SetupRoundFinished,
  PlayingRound,
  RoundFinished,
  GameFinished,
  WaitingForPlayer,
  UnavailableMove,
};

static const char* game_states[] = {
    "UnInitialized",
    "Ready to start",
    "Setup Round",
    "Setup Round Finished",
    "Playing Round",
    "Round Finished",
    "Game Finished",
    "Waiting for player",
    "Unavailable Move"
};

struct GameInfo {
  uint8_t current_dev_card;
  TurnType turn_type;
  int current_round;
};

/******************
 *    Logging     *
 ******************/

enum LogType {
  NoLog,
  MoveLog,
  GameLog,
  BothLogs
};

struct GameSummary {
  unsigned int id;
  uint16_t rounds;
  uint16_t moves_played;
  uint8_t run_time;  // ms
  Color winner;
  uint8_t num_players;
  unsigned int seed;
};

struct Logger {
  LogType type = NoLog;
  FILE* game_summary_file = nullptr;
  FILE* move_file = nullptr;

  GameSummary* game_summaries;
  Move* moves;
  unsigned int writes = 0;
  unsigned int games_played = 0;
};

/*************
 *    AI     *
 *************/

struct AISummary {
  PlayerType type;
  uint16_t id;
  unsigned int seed;
  float win_rate;  // between 0-1
  float average_moves;
  float average_points;  // between 0-11
};


#endif //FASTCATAN_COMPONENTS_H
