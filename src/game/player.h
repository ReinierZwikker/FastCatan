#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H

#include "components.h"
#include "board.h"

class PlayerAgent {
public:
  virtual inline Move get_move_gui(Board *board, int cards[5]) { return {}; }
  virtual inline Move get_move(Board *board, int cards[5]) { return {}; }
  virtual inline void finish_round(Board *board) {}
};

class Player {
public:
  Player(Board *global_board, Color player_color);

  bool activated = false;


  /*  === Link to player agent ===
   * Agent should implement the following function:
   * get_move(board, cards, available_moves)
   * finish_round(board, scores)
   *
   */

  PlayerAgent *agent = nullptr;

  Board *board;

  Move *available_moves;

  Color player_color = NoColor;

  //                      {Streets, Villages, Cities}
  int resources_left[3] = {     14,        5,      4};

  int cards[5]{};

  // TODO evaluate if we want to keep this here
  int place_street(int street_id);
  int place_village(int corner_id);
  int place_city(int corner_id);

  Move *update_available_moves(TurnType turn_type, Player *players[4]);

  virtual ~Player();

  void set_cards(int brick, int lumber, int ore, int grain, int wool);

  bool resources_for_street();
  bool resources_for_village();
  bool resources_for_city();
  bool resources_for_development();
};

#endif //FASTCATAN_PLAYER_H
