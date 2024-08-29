#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H

#include <set>
#include <vector>
#include "components.h"
#include "board.h"

class PlayerAgent {
public:
  virtual inline ~PlayerAgent() = default;
  virtual inline Move get_move(Board *board, int cards[5], GameInfo game_info) { return {}; }
  virtual inline void finish_round(Board *board) {}
  virtual inline PlayerType get_player_type() { return PlayerType::NoPlayer; }
  virtual inline PlayerState get_player_state() { return Waiting; }
  virtual inline void *get_custom_player_attribute() { return nullptr; }
  virtual inline void unpause(Move move) {}
  unsigned int agent_seed;
};

class Player {
public:
  Player(Board *global_board, Color player_color);
  Player(Board *global_board, Color player_color, int given_id);

  Player();

  bool activated = false;

  // Freely assignable Player ID used for player tracking purposes while training
  int player_id = 0;

  /*  === Link to player agent ===
   * Agent should implement the following function:
   * get_move(board, cards, available_moves)
   * finish_round(board, scores)
   * get_player_type()
   */

  PlayerAgent *agent = nullptr;

  Board *board;

  Move *available_moves;

  Color player_color = Color::NoColor;

  //                      {Streets, Villages, Cities}
  int resources_left[3] = {     15,        5,      4};
  int cards[5]{};

  // Harbors                  {Brick, Lumber, Ore,   Grain, Whool}
  bool available_harbors[5] = {false, false,  false, false, false};

  // Development Cards
  std::vector<DevelopmentCard> development_cards{};
  int dev_cards[5]{};
  bool played_development_card = false;
  int victory_cards = 0;

  // Knights
  int played_knight_cards = 0;
  bool knight_leader = false;

  // Longest trade route
  int longest_route = 0;  // The longest route this player has
  bool road_leader = false;  // If this player gets the victory points for longest route

  int victory_points = 0;

  ~Player();

  bool resources_for_street();
  bool resources_for_village();
  bool resources_for_city();
  bool resources_for_development();

  std::set<int> traverse_route(int street_id, std::set<int> previous_streets, std::set<int>* other_route,
                               int previous_corner, Color color);
  void check_longest_route(int street_id, Color color);

  // TODO evaluate if we want to keep this here
  int place_street(int street_id);
  int place_village(int corner_id);
  int place_city(int corner_id);

  Move *update_available_moves(TurnType turn_type, Player *players[4], int current_development_card);

  void set_cards(int brick, int lumber, int ore, int grain, int wool);
  void add_cards(CardType card_type, int amount);
  void remove_cards(CardType card_type, int amount);

  void buy_development(DevelopmentType development_card);
  void play_development(int development_index);
  void activate_development_cards();

  int get_total_amount_of_cards();

  Move *add_new_move(int move_id);

  int check_victory_points();
};

#endif //FASTCATAN_PLAYER_H
