#ifndef FASTCATAN_PLAYER_H
#define FASTCATAN_PLAYER_H

#include "components.h"
#include "board.h"

struct Player {
public:
  Player(Board *global_board, Color player_color);

  bool activated = false;


  /*  === Link to player agent ===
   * Agent should implement the following functions:
   * init_agent(board)
   * first_town(current_game_state)
   * second_town(current_game_state)
   * do_move(current_game_state)
   *
   * Connect using:
   * https://linux.die.net/man/2/pipe
   * https://linux.die.net/man/2/fork
   */

  Move first_town(...);
  Move second_town(...);

  Move do_move(...);

  Board *board;

  Color player_color = NoColor;

  //                      {Streets, Villages, Cities}
  int resources_left[3] = {     14,        5,      4};

  int cards[5]{};

  // TODO evaluate if we want to keep this here
  int place_street(int street_id);
  int place_village(int corner_id);
  int place_city(int corner_id);
};

#endif //FASTCATAN_PLAYER_H
