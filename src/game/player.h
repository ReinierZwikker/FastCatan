#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H

#include <set>
#include <vector>
#include "components.h"
#include "board.h"

class PlayerAgent {
public:
  virtual inline Move get_move(Board *board, int cards[5]) { return {}; }
  virtual inline void finish_round(Board *board) {}
  virtual inline PlayerType get_player_type() { return NoPlayer; }
  virtual inline PlayerState get_player_state() { return Waiting; }
  virtual inline void unpause(Move move) {}
};

class Player {
public:
  Player(Board *global_board, Color player_color);

  bool activated = false;

  /*  === Link to player agent ===
   * Agent should implement the following function:
   * get_move(board, cards, available_moves)
   * finish_round(board, scores)
   * get_player_type()
   */

  PlayerAgent *agent = nullptr;

  Board *board;

  Move *available_moves;

  Color player_color = NoColor;

  //                      {Streets, Villages, Cities}
  int resources_left[3] = {     15,        5,      4};
  int cards[5]{};

  // Development Cards
  std::vector<DevelopmentCard> development_cards{};
  bool played_development_card = false;
  int victory_cards = 0;

  // Knights
  int played_knight_cards = 0;
  bool knight_leader = false;

  // Longest trade route
  int longest_route = 0;  // The longest route this player has
  bool road_leader = false;  // If this player gets the victory points for longest route

  int victory_points = 0;

  virtual ~Player();

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
